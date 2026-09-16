from pathlib import Path
import pandas as pd

SOURCE = Path(r"D:\Hackathons\SIH 2026\data\external\elipticpp")
OUTPUT = Path(r"D:\Hackathons\SIH 2026\final\data\canonical")


TRANSACTION_COLUMNS = [
    "txId", "Time step", "total_BTC", "fees",
    "num_input_addresses", "num_output_addresses",
    "in_BTC_min", "in_BTC_max", "in_BTC_mean", "in_BTC_median", "in_BTC_total",
    "out_BTC_min", "out_BTC_max", "out_BTC_mean", "out_BTC_median", "out_BTC_total",
]


def load_wallet_relationships(path, tx_column, wallet_column):
    df = pd.read_csv(path, usecols=[tx_column, wallet_column])
    groups = df.groupby(tx_column, sort=False)[wallet_column].agg(list)
    del df
    return groups


def build_canonical_transactions():
    OUTPUT.mkdir(parents=True, exist_ok=True)

    tx_features = pd.read_csv(
        SOURCE / "transactions" / "txs_features.csv",
        usecols=TRANSACTION_COLUMNS,
    )

    input_groups = load_wallet_relationships(
        SOURCE / "actor" / "AddrTx_edgelist.csv",
        "txId",
        "input_address",
    )

    output_groups = load_wallet_relationships(
        SOURCE / "actor" / "TxAddr_edgelist.csv",
        "txId",
        "output_address",
    )

    canonical = tx_features.copy()
    del tx_features

    canonical["input_wallets"] = canonical["txId"].map(input_groups)
    canonical["output_wallets"] = canonical["txId"].map(output_groups)
    del input_groups
    del output_groups

    canonical["input_wallets"] = canonical["input_wallets"].apply(
        lambda x: x if isinstance(x, list) else []
    )
    canonical["output_wallets"] = canonical["output_wallets"].apply(
        lambda x: x if isinstance(x, list) else []
    )

    canonical["has_address_relationships"] = (
        canonical["input_wallets"].str.len().gt(0)
        | canonical["output_wallets"].str.len().gt(0)
    )

    canonical = canonical.rename(columns={
        "txId": "txid",
        "Time step": "time_step",
        "total_BTC": "total_btc",
        "fees": "fee_btc",
        "num_input_addresses": "input_address_count",
        "num_output_addresses": "output_address_count",
        "in_BTC_min": "input_btc_min",
        "in_BTC_max": "input_btc_max",
        "in_BTC_mean": "input_btc_mean",
        "in_BTC_median": "input_btc_median",
        "in_BTC_total": "input_btc_total",
        "out_BTC_min": "output_btc_min",
        "out_BTC_max": "output_btc_max",
        "out_BTC_mean": "output_btc_mean",
        "out_BTC_median": "output_btc_median",
        "out_BTC_total": "output_btc_total",
    })

    output_file = OUTPUT / "canonical_transactions.parquet"
    canonical.to_parquet(output_file, index=False)

    print("Rows:", len(canonical))
    print("Unique txid:", canonical["txid"].nunique())
    print("Transactions with address relationships:", int(canonical["has_address_relationships"].sum()))
    print("Transactions without address relationships:", int((~canonical["has_address_relationships"]).sum()))
    print("Created:", output_file)


if __name__ == "__main__":
    build_canonical_transactions()
