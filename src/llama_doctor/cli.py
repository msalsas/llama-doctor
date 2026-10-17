import argparse


def main(argv=None):
    parser = argparse.ArgumentParser(prog="llama-doctor")
    parser.add_subparsers(dest="command")
    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        return 0
