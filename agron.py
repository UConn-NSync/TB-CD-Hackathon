import decimal

def response_fill(price: decimal.Decimal, volume: decimal.Decimal, error_code:str, error_msg: str, unfilled: dict[str, decimal.Decimal]):
    x = dict()
    x['price'] = price
    x['volume'] = volume
    x['error_code'] = error_code
    x['error_msg'] = error_msg
    x['unfilled'] = unfilled
    return x



{
  “price”: decimal.Decimal,       # fill price
  “volume”: decimal.Decimal,      # fill volume
  “error_code”: str               # Optional. “rejected”
  “error_msg”: str                # Optional. description of error
  “unfilled”: dict[str, Decimal]  # unfilled portion of trades (xbt,eth)
}