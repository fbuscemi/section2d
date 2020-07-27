import argparse
import pandas as pd 
import re

# pattern to extract key/value pairs from a block
PAT = re.compile("([a-z]+) = (\d+)")

# convert solution block to dict
def block_to_dict(block):
    items = PAT.findall(block)
    if items:
        return {k: v for k, v in items}
    else:
        return None

# convert minizinc result text to pandas dataframe
def txt_to_df(txt):
    blocks = txt.split("----------")
    d = []
    for block in blocks:
        row = block_to_dict(block)
        if row:
            d.append(row)
    return pd.DataFrame(d)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Convert minizinc output to csv')

    parser.add_argument('--infile', type=str, help='input file')
    parser.add_argument('--outfile', type=str, help='output file')

    args = parser.parse_args()


    with open(args.infile, encoding='utf-8') as fp:
        txt = fp.read()
    df = txt_to_df(txt)
    df.to_csv(args.outfile, index=False)


