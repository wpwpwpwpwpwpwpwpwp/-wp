from datetime import date
from enum import Enum
from logging import getLogger
from typing import *

import pandas as pd

from quant_engine.core import *
from quant_engine.typing_ import *


class TableNames(str, Enum):
    MAIN = 'main'


def select_hermes_columns_from_table(db: SqlDB,
                                     table_name: str,
                                     column_mapping: Sequence[Tuple[str, str]],
                                     start_time: Union[date, str],
                                     end_time: Union[date, str]):

    columns = [f't."{a}" AS "{b}"' for a, b in column_mapping]
    if not columns:
        raise ValueError(f'`column_mapping` must not be empty.')

    stmt = f'''
    SELECT
        t."trade_date" AS "trade_date",
        t2."security_id" AS "datayes_id",
        t2."ticker_symbol" AS "datayes_code",
        t2."sec_short_name" AS "sec_name",
        {", ".join(columns)}
    FROM
        "{table_name}" AS t
    LEFT JOIN
        "md_security" AS t2
    ON
        t.security_id = t2.security_id AND
        t.trade_date >= %(start_time)s AND
        t.trade_date < %(end_time)
    ;
    '''

    with db.scoped_conn() as conn:
        return pd.read_sql(
            sql=stmt,
            con=conn,
            params={'start_time': start_time, 'end_time': end_time},
        )


def merge_hermes_daily_data(db: SqlDB,
                            start_time: Union[date, str],
                            end_time: Union[date, str]):
    table_column_mappings = [
        ('mkt_equd', [
            ('pre_close_price', 'exchange_pre_close_adj'),
            ('act_pre_close_price', 'pre_close'),
            ('highest_price', 'high'),
            ('lowest_price', 'low'),
            ('close_price', 'close'),
            ('turnover_vol', 'volume'),
            ('turnover_value', 'amount'),
            ('deal_amount', 'deal_num'),
            ('neg_market_value', 'mkt_freeshares'),
            ('market_value', 'mkt_cap_ard'),
            ('chg_pct', 'pct_chg'),
            ('turnover_rate', 'turn'),
        ]),
        ('mkt_equd_adj', [
            ('pre_close_price_1', 'pre_close_adj'),
            ('open_price_1', 'open_adj'),
            ('high_price_1', 'high_adj'),
            ('low_price_1', 'low_adj'),
            ('close_price_1', 'close_adj'),
            ('turnover_vol', 'volume_adj'),
        ]),
        ('mkt_equd_eval', [
            ('pe_t', 'pe_ttm'),
            ('pe_m', 'pe_mrq'),
            ('pe_cm', 'pe_deducted_mrq'),
            ('pb', 'pb_lf'),
            ('ps_t', 'ps_ttm'),
            ('ps_m', 'ps_mrq'),
            ('pcf_t', 'pcf_ncf_ttm'),
            ('pcf_m', 'pcf_ncf_mrq'),
            ('pcf_ot', 'pcf_ocf_ttm'),
            ('pcf_om', 'pcf_ocf_mrq'),
            ('ev', 'ev'),
            ('ev_ebitda', 'ev_2_ebitda'),
            ('ev_sales', 'ev_2_sales_ttm'),
        ]),
        ('mkt_equd_ind', [
            ('chg_status', 'trade_status'),
            ('con_chg', 'con_chg'),
            ('con_limit', 'con_limit'),
            ('days_1', 'days_2_highest'),
            ('days_2', 'days_new_highest'),
            ('days_3', 'days_2_lowest'),
            ('days_4', 'days_new_lowest'),
            ('deal_value', 'deal_value'),
        ]),
        ('mkt_limit', [
            ('limit_up_price', 'limit_up'),
            ('limit_down_price', 'limit_down'),
            ('up_limit_reached_times', 'up_limit_reached_times'),
            ('down_limit_reached_times', 'down_limit_reached_times'),
        ]),
        ('mkt_eqy_vwapd', [
            ('vwap', 'vwap'),
        ])
    ]

    data_frames = []
    for table_name, column_mapping in table_column_mappings:
        df = select_hermes_columns_from_table(
            db=db,
            table_name=table_name,
            column_mapping=column_mapping,
            start_time=start_time,
            end_time=end_time,
        )
        # ...
        df.set_index(['code', 'datetime'])
        data_frames.append(df)

    ret_df = pd.concat(data_frames, axis=1)
    ret_df.reset_index()
    return ret_df


class App(ModuleApp):

    def process(self, queries: List[Query]):
        db = self.ctx.databases['hermes_data']
        for q in queries:
            getLogger(__name__).info('received query: %s', q)
            if q.code_set.lower() not in ('', '*', 'all'):
                raise RuntimeError('This module can only be operated on all codes.')

            df = merge_hermes_daily_data(db, q.start_time, q.end_time)
            self.s.write_table(
                str(TableNames.MAIN),
                data=df,
                layout=DataTableLayout.CODE_DATE
            )

    def train_model(self, query: Query):
        pass

    def purge_cache(self, code_sets: Optional[List[str]] = None):
        for table_name in TableNames.__members__:
            self.s.purge_table(str(table_name), code_sets)


if __name__ == '__main__':
    run_module_app(App)
