# AMD PACE Core Functions

These are the core helper functions of AMD PACE. They are not arithmetic ops in the traditional sense; they expose process-level utilities (thread binding, logging) to Python.

The methods are registered as torch dispatcher ops on the `pace` namespace in [`csrc/core/core_ops.cpp`](../csrc/core/core_ops.cpp). At runtime they are reached via `torch.ops.pace.*`, and the `pace.core` Python module ([`pace/core.py`](../pace/core.py)) wraps them with familiar names so existing call sites keep working.

The methods are listed below.

### thread_bind
This method is implemented with the help of `pthread_setaffinity_np` and provides an API to python to bind the calling thread to a specific core or a set of cores.
* Operation: `pace.core.thread_bind` (dispatcher op: `torch.ops.pace.thread_bind`)
* Arguments:
    * `int[] core_ids`: List of core ids to bind the thread to.
* Files: `csrc/core/threading.cpp` (implementation), `csrc/core/core_ops.cpp` (registration)
* Example usage:
    ```python
    import torch
    import pace
    from multiprocessing import Process

    def f():
        pace.core.thread_bind([0, 1, 2, 3])
        print("Thread bound to cores 0, 1, 2, 3")

    p = Process(target=f)
    p.start()
    p.join()
    ```

## pace_logger
This method provides an API to python to log messages to the console.
* Operation: `pace.core.pace_logger` (dispatcher op: `torch.ops.pace.log`)
* Arguments:
    * `int level`: Log level. Can be one of the following:
        * `0`: DEBUG
        * `1`: PROFILE
        * `2`: INFO
        * `3`: WARNING
        * `4`: ERROR
    * `str message`: Message to be logged.
* Files: `csrc/core/logging.h` (implementation), `csrc/core/core_ops.cpp` (registration)
* Example usage: This method is (ideally) not to be used directly by the user, but is used internally by the python library to log messages. A helper method is provided as part of utils and can be used by

    ```
    from pace.utils import pacelogger, logLevel

    pacelogger(logLevel.INFO, message)
    ```

The logging can be controlled by setting the environment variable `PACE_LOG_LEVEL`. Refer to [README](../README.md#verbose) for more details.

More information for developers can be found [here](./Contributing.md#logging-in-amd-pace)
