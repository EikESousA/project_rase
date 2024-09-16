from datetime import datetime


def print_log(
    finish: bool,
    module: str,
    action: str,
    message: str,
    end: str | None = "\n",
    clean: bool | None = False,
) -> None:
    """
    Impressão de log.

    Args:
        finish (bool): Finalizou a acao.
        module (str): Modulo do log.
        action (str): Acao do log.
        message (str): Mensagem do log.

    Returns:

    """

    if clean:
        print("\b", end=end)
    else:
        date = datetime.now().strftime("%d-%m-%Y %H:%M:%S")
        print(
            f"| {date} | {'[x]' if finish else '[-]'} | {module} | {action} | {message}",
            end=end,
        )
