"""Asigna un lote concreto a una venta confirmada, sin modificar el comprobante.

Uso: python scripts/assign_sale_lot.py --sale-id 19 --buy-id 18 --apply
Sin --apply solo inspecciona. La migración hace una copia SQLite antes de escribir.
"""

from __future__ import annotations

import argparse
from decimal import Decimal
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from portfolio_tracker.config import DB_PATH, LOCAL_TIMEZONE
from portfolio_tracker.db import Database
from portfolio_tracker.repository import PortfolioRepository
from portfolio_tracker.services.lot_sale_simulator import open_buy_lots
from portfolio_tracker.services.realized_performance import realized_sales


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sale-id", type=int, required=True)
    parser.add_argument("--buy-id", type=int, required=True)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()

    database = Database(DB_PATH)
    if not args.apply and database.schema_version() < 14:
        print("Vista previa: falta migración v14; no se escribió nada.")
        return
    if args.apply:
        database.initialize()
    repository = PortfolioRepository(database)
    trades = repository.list_trades(ascending=True)
    by_id = {int(row["id"]): row for row in trades}
    sale = by_id.get(args.sale_id)
    buy = by_id.get(args.buy_id)
    if sale is None or buy is None:
        raise ValueError("Operación no encontrada; no se asignó nada.")
    print("Venta", sale["id"], sale["symbol"], sale["side"], sale["quantity"],
          sale["price_usd"], "comisión", sale["commission_usd"])
    print("Compra", buy["id"], buy["symbol"], buy["side"], buy["quantity"],
          buy["price_usd"], "comisión", buy["commission_usd"])
    if args.apply:
        if not (sale["symbol"] == buy["symbol"] == "SMCI"
                and sale["side"] == "SELL" and buy["side"] == "BUY"
                and Decimal(sale["quantity"]) == Decimal(buy["quantity"]) == Decimal("5")
                and Decimal(sale["price_usd"]) == Decimal("33")
                and Decimal(buy["price_usd"]) == Decimal("32.56")
                and Decimal(sale["commission_usd"]) == Decimal("0.41")
                and Decimal(buy["commission_usd"]) == Decimal("0.40")):
            raise ValueError("Los comprobantes no coinciden con la corrección autorizada.")
        created = repository.assign_sale_to_buy_lot(
            sale_trade_id=args.sale_id, buy_trade_id=args.buy_id,
            reason="Usuario confirmó que esta venta corresponde a la compra reciente de cinco acciones",
        )
        print("Asignación creada:", created)
        trades = repository.list_trades(ascending=True)
    result = next((item for item in realized_sales(trades, timezone=LOCAL_TIMEZONE)
                   if item.trade_id == args.sale_id), None)
    print("Resultado de venta:", result.gain_usd if result else "N/D")
    print("Lotes SMCI abiertos:", [
        (item.trade_id, str(item.remaining_quantity))
        for item in open_buy_lots(trades) if item.symbol == "SMCI"
    ])
    print("Integridad SQLite:", database.integrity_check())


if __name__ == "__main__":
    main()
