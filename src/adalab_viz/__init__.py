from .cli import main

try:
    import matplotlib
    import seaborn
    import mplfonts
except ImportError:
    raise ImportError(
        "adalab_viz requires visualization dependencies.\n"
        "Install them via: pip install 'adalab[viz]'"
    )
if __name__ == "__main__":
    main()
