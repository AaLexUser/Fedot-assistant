import asyncio
import logging
from datetime import datetime
from typing import Any, Callable, Coroutine, TypeAlias

from fedotllm.server.event import Event
from enum import Enum

logger = logging.getLogger(__name__)

AsyncEventHandler: TypeAlias = Callable[[Event], Coroutine[Any, Any, None]]


class EventStreamSubscriber(str, Enum):
    # Enum definition remains the same
    AGENT_CONTROLLER = "agent_controller"
    SECURITY_ANALYZER = "security_analyzer"
    RESOLVER = "openhands_resolver"
    SERVER = "server"
    RUNTIME = "runtime"
    MEMORY = "memory"
    MAIN = "main"
    TEST = "test"


class EventStream:
    sid: str
    _queue: asyncio.Queue[Event | None]  # Use asyncio Queue, allow None for sentinel
    _subscribers: dict[
        str, dict[str, AsyncEventHandler]
    ]  # Subscriber ID -> Callback ID -> Callback
    _lock: asyncio.Lock  # Async lock for cur_id
    _process_task: asyncio.Task | None  # Task for processing the queue
    _close_timeout: float  # Configurable shutdown timeout
    _cur_id: int

    def __init__(self, sid: str, close_timeout: float = 10.0) -> "EventStream":
        self.sid = sid
        self._queue = asyncio.Queue()
        self._subscribers = {}
        self._lock = asyncio.Lock()
        self._process_task = None
        self._close_timeout = close_timeout
        self._cur_id = 0

    async def start_processing(self):
        """Starts the background task that processes events from the queue."""
        if self._process_task is None or self._process_task.done():
            logger.info(f"Starting event processing task for sid={self.sid}")
            self._process_task = asyncio.create_task(
                self._process_queue(), name=f"EventStreamProcessor-{self.sid}"
            )
            # Add a callback to log errors if the task crashes
            self._process_task.add_done_callback(self._log_task_completion)
        else:
            logger.warning(f"Processing task for sid={self.sid} is already running.")

    async def close(self) -> None:
        """Gracefully shuts down the event stream processing."""
        logger.info(f"Closing event stream for sid={self.sid}...")

        # Signal the processing loop to stop by putting a sentinel value (None)
        if self._process_task and not self._process_task.done():
            await self._queue.put(None)  # Sentinel value to stop the processor

            try:
                # Wait for the processing task to complete with a timeout
                await asyncio.wait_for(self._process_task, timeout=self._close_timeout)
                logger.info(
                    f"Event stream processor for sid={self.sid} stopped gracefully"
                )
            except asyncio.TimeoutError:
                logger.warning(
                    f"Timeout waiting for event processor to stop for sid={self.sid}. Cancelling task."
                )
                self._process_task.cancel()
                try:
                    await self._process_task  # Wait for cancellation to complete
                except asyncio.CancelledError:
                    logger.info(
                        f"Event processor task for sid={self.sid} cancelled successfully"
                    )
                except Exception as e:
                    logger.error(
                        f"Error during task cancellation for sid={self.sid}: {e}",
                        exc_info=True,
                    )
            except Exception as e:
                logger.error(
                    f"Error during event stream shutdown for sid={self.sid}: {e}",
                    exc_info=True,
                )

        # Clear all subscribers to prevent any more callbacks
        self._subscribers.clear()
        logger.info(f"Event stream for sid={self.sid} closed successfully")

    def subscribe(
        self,
        subscriber_id: EventStreamSubscriber,
        callback: AsyncEventHandler,  # Use TypeAlias
        callback_id: str,
    ) -> None:
        """
        Subscribes an asynchronous callback to the event stream.
        Raises TypeError if the callback is not an async function.
        """
        # Enforce async callback type
        if not asyncio.iscoroutinefunction(callback):
            raise TypeError(
                f"Callback for {subscriber_id}/{callback_id} must be an async function (defined with 'async def')."
            )

        logger.debug(f"Subscribing {subscriber_id}/{callback_id} for sid={self.sid}")
        if subscriber_id not in self._subscribers:
            self._subscribers[subscriber_id] = {}

        if callback_id in self._subscribers[subscriber_id]:
            logger.warning(
                f"Callback ID '{callback_id}' on subscriber '{subscriber_id}' for sid={self.sid} already exists. Overwriting."
            )
            # Or raise ValueError if overwriting is not desired:
            # raise ValueError(f"Callback ID '{callback_id}' on subscriber '{subscriber_id}' already exists.")

        self._subscribers[subscriber_id][callback_id] = callback

    async def add_event(self, event: Event) -> None:
        """
        Adds an event to the stream, assigns an ID, persists it (asynchronously),
        and puts it on the queue for processing.

        Raises:
            ValueError: If the event already has an ID.
            Exception: If persistence to the FileStore fails (propagated from file_store.write).
                       The event will NOT be queued if persistence fails.
        """
        if event.id != Event.INVALID_ID:
            logger.error(
                f"Event already has an ID: {event.id} for sid={self.sid}. Possible duplicate add."
            )
            raise ValueError(f"Event already has an ID:{event.id}. Cannot add again.")
        # Assign ID atomically
        async with self._lock:
            event._id = self._cur_id
            self._cur_id += 1
        event._timestamp = datetime.now().isoformat()

        await self._queue.put(event)

    def unsubscribe(
        self, subscriber_id: EventStreamSubscriber, callback_id: str
    ) -> None:
        """Unsubscribes a callback."""
        logger.debug(f"Unsubscribing {subscriber_id}/{callback_id} for sid={self.sid}")
        if (
            subscriber_id in self._subscribers
            and callback_id in self._subscribers[subscriber_id]
        ):
            del self._subscribers[subscriber_id][callback_id]
            # Remove subscriber_id from dict if empty
            if not self._subscribers[subscriber_id]:
                del self._subscribers[subscriber_id]
        else:
            logger.warning(
                f"Attempted to unsubscribe non-existent callback: {subscriber_id}/{callback_id} for sid={self.sid}"
            )

    async def _process_queue(self) -> None:
        """Continuously processes events from the queue and calls subscribers."""
        logger.info(f"Event processing loop started for sid={self.sid}.")
        while True:
            event = await self._queue.get()

            # Sentinel value (None) used to signal shutdown
            if event is None:
                logger.info(
                    f"Sentinel received, stopping event processing loop for sid={self.sid}."
                )
                self._queue.task_done()
                break

            logger.debug(
                f"Processing event id={event.id} ({type(event).__name__}) for sid={self.sid}"
            )
            # Use asyncio.gather to run subscriber callbacks concurrently
            tasks = []
            for subscriber_id in sorted(
                self._subscribers.keys()
            ):  # Sort for consistent order (optional)
                callbacks = self._subscribers[subscriber_id]
                for callback_id, callback in callbacks.items():
                    # Create a task for each callback
                    task_name = f"Callback-{subscriber_id}-{callback_id}-Event-{event.id}-sid-{self.sid}"
                    tasks.append(
                        asyncio.create_task(
                            self._run_callback(
                                callback, event, callback_id, subscriber_id
                            ),
                            name=task_name,
                        )
                    )

            # Wait for all callbacks for this event to complete
            if tasks:
                # results = await asyncio.gather(*tasks, return_exceptions=True) # Use return_exceptions=True if you want to handle errors here
                await asyncio.gather(
                    *tasks
                )  # Exceptions are logged within _run_callback

            # Mark the event as processed in the queue
            self._queue.task_done()

        logger.info(f"Event processing loop finished for sid={self.sid}.")

    async def _run_callback(
        self,
        callback: AsyncEventHandler,
        event: Event,
        callback_id: str,
        subscriber_id: str,
    ):
        """Safely runs a single subscriber callback."""
        try:
            logger.debug(
                f"Calling callback {subscriber_id}/{callback_id} for event id={event.id}, sid={self.sid}"
            )
            await callback(event)
        except Exception as e:
            logger.error(
                f"Error in event callback {callback_id} for subscriber {subscriber_id} processing event id={event.id}, sid={self.sid}: {e}",
                exc_info=True,  # Log the full traceback
            )

    def _log_task_completion(self, task: asyncio.Task) -> None:
        """Callback added to the main processing task to log exceptions."""
        try:
            # If the task raised an exception, accessing result() will re-raise it
            task.result()
            logger.info(
                f"Event processing task {task.get_name()} completed successfully."
            )
        except asyncio.CancelledError:
            logger.info(f"Event processing task {task.get_name()} was cancelled.")
        except Exception as e:
            # Log the exception from the task if it failed unexpectedly
            logger.error(
                f"Event processing task {task.get_name()} failed unexpectedly:",
                exc_info=e,
            )
