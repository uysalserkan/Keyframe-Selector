"""
Timing and profiling utilities.
"""

import functools
import logging
import time
from contextlib import contextmanager
from typing import Any, Callable, Dict, Optional

logger = logging.getLogger(__name__)


class Timer:
    """
    Context manager and decorator for timing code execution.
    
    Usage as context manager:
        with Timer("my_operation") as t:
            do_something()
        print(f"Took {t.elapsed:.3f}s")
    
    Usage as decorator:
        @Timer.decorate("my_function")
        def my_function():
            ...
    """
    
    _timings: Dict[str, float] = {}
    
    def __init__(self, name: str = "operation", log: bool = True):
        """
        Initialize timer.
        
        Args:
            name: Name of the operation being timed.
            log: Whether to log the timing result.
        """
        self.name = name
        self.log = log
        self.start_time: float = 0.0
        self.end_time: float = 0.0
        self.elapsed: float = 0.0
    
    def __enter__(self) -> "Timer":
        self.start_time = time.perf_counter()
        return self
    
    def __exit__(self, *args) -> None:
        self.end_time = time.perf_counter()
        self.elapsed = self.end_time - self.start_time
        Timer._timings[self.name] = self.elapsed
        
        if self.log:
            logger.info(f"[TIMER] {self.name}: {self.elapsed:.3f}s")
    
    @classmethod
    def decorate(cls, name: Optional[str] = None, log: bool = True) -> Callable:
        """
        Decorator factory for timing functions.
        
        Args:
            name: Custom name (defaults to function name).
            log: Whether to log the timing.
        
        Returns:
            Decorator function.
        """
        def decorator(func: Callable) -> Callable:
            timer_name = name or func.__name__
            
            @functools.wraps(func)
            def wrapper(*args, **kwargs) -> Any:
                with cls(timer_name, log=log):
                    return func(*args, **kwargs)
            
            return wrapper
        return decorator
    
    @classmethod
    def get_timings(cls) -> Dict[str, float]:
        """Get all recorded timings."""
        return cls._timings.copy()
    
    @classmethod
    def clear_timings(cls) -> None:
        """Clear all recorded timings."""
        cls._timings.clear()
    
    @classmethod
    def summary(cls) -> str:
        """Get a summary of all timings."""
        if not cls._timings:
            return "No timings recorded."
        
        lines = ["Timing Summary:", "-" * 40]
        total = 0.0
        for name, elapsed in sorted(cls._timings.items()):
            lines.append(f"  {name}: {elapsed:.3f}s")
            total += elapsed
        lines.append("-" * 40)
        lines.append(f"  Total: {total:.3f}s")
        return "\n".join(lines)


def profile_function(func: Callable) -> Callable:
    """
    Simple decorator to profile a function's execution time.
    
    Args:
        func: Function to profile.
    
    Returns:
        Wrapped function that logs execution time.
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs) -> Any:
        start = time.perf_counter()
        try:
            result = func(*args, **kwargs)
            return result
        finally:
            elapsed = time.perf_counter() - start
            logger.debug(f"{func.__name__} took {elapsed:.4f}s")
    
    return wrapper


@contextmanager
def time_block(name: str):
    """
    Simple context manager for timing a code block.
    
    Args:
        name: Name of the code block.
    
    Yields:
        Dictionary that will contain 'elapsed' after the block.
    """
    result = {"elapsed": 0.0}
    start = time.perf_counter()
    try:
        yield result
    finally:
        result["elapsed"] = time.perf_counter() - start
        logger.debug(f"{name}: {result['elapsed']:.4f}s")
