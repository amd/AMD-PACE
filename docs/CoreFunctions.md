# AMD PACE Core Functions

These are the core functions of AMD PACE, which are not essentially ops but are helper functions to perform various tasks. These are not registered as ops in the `torch` library, but are available in the `pace` library.

The methods are registered in the library using `PYBIND11_MODULE` in `csrc/init.cpp`, which delegates to `csrc/torch_extension_bindings.cpp`.

The methods are listed below.

### thread_bind
This method is implemented with the help of `pthread_setaffinity_np` function and provides an API to python to bind the calling thread to a specific core or a set of cores.
* Operation: `pace.core.thread_bind`
* Arguments:
    * `List core_ids`: List of core ids to bind the thread to.
* File: `csrc/core/threading.cpp`
* Example usage:
    ```python
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
* Operation: `pace.core.pace_logger`
* Arguments:
    * `int level`: Log level. Can be one of the following:
        * `0`: DEBUG
        * `1`: PROFILE
        * `2`: INFO
        * `3`: WARNING
        * `4`: ERROR
    * `String message`: Message to be logged.
* File: `csrc/core/logging.cpp`
* Example usage: This method is (ideally) not to be used directly by the user, but is used internally by the python library to log messages. A helper method is provided as part of utils and can be used by

    ```
    from pace.utils import pacelogger, logLevel

    pacelogger(logLevel.INFO, message)
    ```

The logging can be controlled by setting the environment variable `PACE_LOG_LEVEL`. Refer to [README](../README.md#verbose) for more details.

More information for developers can be found [here](./Contributing.md#logging-in-amd-pace)
