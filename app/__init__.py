import sys

from app.services.exit_service import exit_service
from app.services.inference_service import inference_service
from app.services.instruction_service import instruction_service
from app.services.rase_service import rase_service
from app.utils.log.print_log import print_log


def app() -> None:
    """Digitar"""

    try:
        print_log(
            finish=False, module="SYST", action="MENU", message="|----------------------------------------------|"
        )
        print_log(
            finish=False, module="SYST", action="MENU", message="| Escolha uma das opções abaixo:               |"
        )
        print_log(
            finish=False, module="SYST", action="MENU", message="|----------------------------------------------|"
        )
        print_log(
            finish=False, module="SYST", action="MENU", message="| 1 - Treinar modelo para geração de texto     |"
        )
        print_log(
            finish=False, module="SYST", action="MENU", message="| 2 - Treinar modelo para RASE                 |"
        )
        print_log(
            finish=False, module="SYST", action="MENU", message="| 3 - Realizar inferência de texto             |"
        )
        print_log(
            finish=False, module="SYST", action="MENU", message="| 4 - Sair                                     |"
        )
        print_log(
            finish=False, module="SYST", action="MENU", message="|----------------------------------------------|"
        )
        answer = input(" Digite o número da opção desejada: ")

        match answer:
            case "1":
                return instruction_service()
            case "2":
                return rase_service()
            case "3":
                return inference_service()
            case "4":
                print_log(finish=False, module="SYST", action="EXIT", message="Encerrado.")
                return exit_service()
            case _:
                print_log(finish=False, module="SYST", action="EXIT", message="Opção não identificada.")
                print_log(finish=False, module="SYST", action="EXIT", message="Encerrado.")
                return exit_service()

    except KeyboardInterrupt:
        print_log(finish=False, module="SYST", action="EXIT", message="Encerrado.")

        sys.exit(0)
