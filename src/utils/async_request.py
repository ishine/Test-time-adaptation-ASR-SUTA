import asyncio
import threading
import traceback
import time


class AsyncIOThread(threading.Thread):
    def __init__(self, loop: asyncio.AbstractEventLoop):
        super().__init__()
        self._loop = loop
        self.daemon = True

    def run(self) -> None:
        self._loop.run_forever()


class OrderPreservedAsyncRequestHandler(object):
    def __init__(self, max_req: int=20):
        self._loop = asyncio.new_event_loop()
        thd = AsyncIOThread(self._loop)
        thd.start()

        self._is_running = False
        self.req_cnt = 0
        self.res_cnt = 0
        self.pending_requests = {}
        self.request_queue = asyncio.Queue()
        self.is_pending = asyncio.Event()
        self.is_not_busy = asyncio.Event()
        self.is_not_busy.set()
        self.is_all_done = asyncio.Event()
        self.is_all_done.set()

        self.max_req = max_req

    async def _request_loop(self):
        while True:
            await self.is_not_busy.wait()
            req = await self.request_queue.get()
            if req is None:  # close signal
                self.close_cnt = self.req_cnt
                continue
            # print(f"Send request ({self.req_cnt}):")
            self.pending_requests[self.req_cnt] = asyncio.create_task(self._wrap_request(req))
            self.req_cnt += 1  # ensure ordering
            if self.req_cnt > self.res_cnt:
                # print("pend!")
                self.is_pending.set()
                self.is_all_done.clear()

            # control maximum requests
            if self.req_cnt >= self.res_cnt + self.max_req:
                self.is_not_busy.clear()

            # wait
            if self.req_cnt % 100 == 0:
                # print("Wait for: ", self.req_cnt)
                await self.is_all_done.wait()
                # print("Completed.")

    async def _response_loop(self):
        while True:
            await self.is_pending.wait()  # check if there exists any request under pending
            try:
                res = await self.pending_requests[self.res_cnt]
                # print(f"Receive request ({self.res_cnt})")
                await self._postprocess(res)
            except:  # request is dead due to reset()
                raise
            self.res_cnt += 1  # ensure ordering
            if self.res_cnt == self.req_cnt:
                self.is_pending.clear()
                self.is_all_done.set()
            
            # control maximum requests
            if self.req_cnt < self.res_cnt + self.max_req:
                self.is_not_busy.set()

            # close
            if self.res_cnt == self.close_cnt:
                print("Gracefully complete all requests!")
                break
    
    def run(self, wrap_request_func, postprocess_func):
        self._is_running = True
        self.close_cnt = -1
        self._wrap_request = wrap_request_func
        self._postprocess = postprocess_func
        async def f():
            coros = [
                self._request_loop(),
                self._response_loop(),
            ]
            tasks = [asyncio.create_task(coro) for coro in coros]
            await run_parallel_tasks(tasks)
            self._is_running = False
        self.fut = asyncio.run_coroutine_threadsafe(f(), self._loop)
    
    def is_running(self) -> bool:
        return self._is_running
    
    def request(self, req):
        async def f():
            await self.request_queue.put(req)
        asyncio.run_coroutine_threadsafe(f(), self._loop)

    def close(self):
        assert self.is_running()
        async def f():
            await self.request_queue.put(None)
        asyncio.run_coroutine_threadsafe(f(), self._loop)
        while True:
            if self.fut.done():
                break
            time.sleep(1)


# exception handling
def handle_asyncio_exception(fut: asyncio.Task):
    # check if the task had an exception
    if fut.exception():
        try:
            # re-raise exception
            fut.result()
        except Exception as e:
            traceback.print_exc()


async def run_parallel_tasks(tasks: list[asyncio.Task]):
    """ Stop all tasks if any of them was successful or raised an exception. Exception is handled. """
    done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
    for task in done:
        handle_asyncio_exception(task)

    for task in pending:
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass


class OrderPreservedTaskManager(object):
    def __init__(self) -> None:
        self.tasks = {}
        self.pending_tasks = {}
        self.req_cnt = 0
        self.res_cnt = 0

        self.is_pending = asyncio.Event()

    def enroll(self, task: asyncio.Task) -> int:
        idx = self.req_cnt
        self.tasks[idx] = task
        self.pending_tasks[idx] = asyncio.get_event_loop().create_future()
        self.req_cnt += 1
        if self.req_cnt > self.res_cnt:
            # print("pend!")
            self.is_pending.set()

        return idx

    async def _response_loop(self):
        while True:
            await self.is_pending.wait()  # check if there exists any request under pending
            try:
                self.pending_tasks[self.res_cnt].set_result(True)
                del self.pending_tasks[self.res_cnt]  # Remove from pending tasks
                # print("Request done: ", self.res_cnt)
            except:
                raise
            self.res_cnt += 1  # ensure ordering
            if self.res_cnt == self.req_cnt:
                self.is_pending.clear()
                self.is_all_done.set()
            
            # control maximum requests
            if self.req_cnt < self.res_cnt + self.max_req:
                self.is_not_busy.set()

            # close
            if self.res_cnt == self.close_cnt:
                print("Gracefully complete all requests!")
                break

    async def wait(self, idx: int):
        await self.pending_tasks[idx]
        return await self.tasks[idx]


if __name__ == '__main__':
    loop = asyncio.new_event_loop()
    thd = AsyncIOThread(loop)
    thd.start()
