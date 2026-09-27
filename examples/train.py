"""Train an algorithm."""

import argparse
import json

from harl.utils.configs_tools import (
    get_defaults_yaml_args,
    update_args,
)


def main():
    """Main function."""

    parser = argparse.ArgumentParser(
        formatter_class=(
            argparse.ArgumentDefaultsHelpFormatter
        )
    )

    parser.add_argument(
        "--algo",
        type=str,
        default="happo",
        choices=["happo"],
        help="Algorithm for cascade reservoir scheduling.",
    )

    parser.add_argument(
        "--env",
        type=str,
        default="cascade_reservoir",
        choices=["cascade_reservoir"],
        help="Five-reservoir cascade environment.",
    )

    parser.add_argument(
        "--exp_name",
        type=str,
        default="installtest",
        help="Experiment name.",
    )

    parser.add_argument(
        "--load_config",
        type=str,
        default="",
        help=(
            "If set, load existing experiment "
            "config file instead of reading from "
            "yaml config file."
        ),
    )

    args, unparsed_args = (
        parser.parse_known_args()
    )

    def process(arg):
        try:
            return eval(arg)
        except Exception:
            return arg

    keys = [
        key[2:]
        for key
        in unparsed_args[0::2]
    ]

    values = [
        process(value)
        for value
        in unparsed_args[1::2]
    ]

    unparsed_dict = {
        key: value
        for key, value
        in zip(
            keys,
            values,
        )
    }

    args = vars(
        args
    )

    if args["load_config"] != "":
        with open(
            args["load_config"],
            encoding="utf-8",
        ) as file:
            all_config = json.load(
                file
            )

        args["algo"] = (
            all_config[
                "main_args"
            ][
                "algo"
            ]
        )

        args["env"] = (
            all_config[
                "main_args"
            ][
                "env"
            ]
        )

        algo_args = (
            all_config[
                "algo_args"
            ]
        )

        env_args = (
            all_config[
                "env_args"
            ]
        )

    else:
        (
            algo_args,
            env_args,
        ) = get_defaults_yaml_args(
            args["algo"],
            args["env"],
        )

    update_args(
        unparsed_dict,
        algo_args,
        env_args,
    )

    if args["env"] != "cascade_reservoir" or args["algo"] != "happo":
        raise ValueError("This repository supports cascade_reservoir with happo only")

    from harl.runners.cascade_reservoir_runner import CascadeReservoirRunner

    runner = CascadeReservoirRunner(args, algo_args, env_args)

    runner.run()

    runner.close()


if __name__ == "__main__":
    main()
    