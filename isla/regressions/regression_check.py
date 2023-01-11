#!/usr/bin/env python
import re
import sys
import pathlib
import argparse

def read_results(p):
    """ given a path to a results txt file produce a dictionary of {"test name": "allow|forbid|error|crash|empty", ...}
    """
    results = {}

    with p.open() as f:
        for line in f:
            m = re.fullmatch("(?P<name>.+),(?P<outcome>.+)\n", line)
            if m is not None:
                results[m.group("name")] = m.group("outcome")

    return results

def produce_table(results):
    all_tests = set(
        t
        for test_results in results.values()
        for t in test_results
    )

    # get keys, and always in that order
    compares = sorted(results.keys())

    table = []

    # header
    table.append(["Test"] + compares)

    for t in sorted(all_tests):
        row = [t]
        for c in compares:
            row.append(
                results[c].get(t, "X")
            )
        table.append(row)

    return table

def format_table(table):
    """ pretty-prints a table built by produce_table into a string
    """
    header = table[0]
    results = table[1:]
    ws = "."*4
    ws_header = " "*4

    maxcolwidth = {
        colidx: max(
            len(row[colidx])
            for row in table
        )
        for colidx in range(len(table[0]))
    }

    lines = []

    line = []
    for cidx, v in enumerate(header):
        mcw = maxcolwidth[cidx]
        line.append(f"{v: <{mcw}}")
    lines.append(ws_header.join(line))

    for row in results:
        line = []
        for cidx, v in enumerate(row):
            mcw = maxcolwidth[cidx]
            line.append(f"{v:.<{mcw}}")
        lines.append(ws.join(line))

    return "\n".join(lines)

def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("compares", nargs="*", type=pathlib.Path)
    args = parser.parse_args(argv)
    results = {}
    for p in args.compares:
        results[p.parent.stem] = read_results(p)
    table = produce_table(results)
    fmt = format_table(table)
    print(fmt)
    return 0

if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
