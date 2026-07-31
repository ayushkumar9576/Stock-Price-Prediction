from rich import print
import time

def starting(name: str) -> float:
    print(f"[bold green]Started {name}[/bold green]")
    return time.perf_counter()


def completion(name: str, start_time: float) -> None:
    elapsed = time.perf_counter() - start_time

    if elapsed < 60:
        duration = f"{elapsed:.2f} sec"
    else:
        duration = f"{int(elapsed // 60)} min {elapsed % 60:.2f} sec"

    print(f"[bold green]Completed {name} ({duration})[/bold green]")