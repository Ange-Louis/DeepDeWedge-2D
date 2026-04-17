import typer

from .fit_model2 import fit_model
from .prepare_data2 import prepare_data
from .refine_tomogram2 import refine_tomogram

# pretty_exceptions_show_locals=False gives shorter error messages
app = typer.Typer(pretty_exceptions_show_locals=False)
app.command()(prepare_data)
app.command()(fit_model)
app.command()(refine_tomogram)


def main():
    app()


if __name__ == "__main__":
    main()
