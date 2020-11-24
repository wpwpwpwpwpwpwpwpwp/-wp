from datetime import date, datetime
from enum import Enum
from logging import getLogger
from typing import *

import pandas as pd

from quant_engine.core import *
from quant_engine.typing_ import *
from quant_engine.utils import *

from lib.hermes import select_hermes_columns_from_table, get_industry_info_from_hermes


class TableNames(BaseTableName):
    DAILY = 'main_daily'
    INDUSTRY = 'main_industry'


def merge_hermes_daily_data(db: SqlDB,
                            datayes_id_list: List[str],
                            start_time: Union[date, str],
                            end_time: Union[date, str]):
    table_column_mappings = [
        ('mkt_equd',
         [('PRE_CLOSE_PRICE', 'exchange_pre_close_adj'),
          ('ACT_PRE_CLOSE_PRICE', 'pre_close'),
          ('OPEN_PRICE', 'open'),
          ('HIGHEST_PRICE', 'high'),
          ('LOWEST_PRICE', 'low'),
          ('CLOSE_PRICE', 'close'),
          ('TURNOVER_VOL', 'volume'),
          ('TURNOVER_VALUE', 'amount'),
          ('DEAL_AMOUNT', 'deal_num'),
          ('NEG_MARKET_VALUE', 'mkt_freeshares'),
          ('MARKET_VALUE', 'mkt_cap_ard'),
          ('CHG_PCT', 'pct_chg'),
          ('TURNOVER_RATE', 'turn')]),
        ('mkt_equd_adj',
         [('PRE_CLOSE_PRICE_1', 'pre_close_adj'),
          ('OPEN_PRICE_1', 'open_adj'),
          ('HIGHEST_PRICE_1', 'high_adj'),
          ('LOWEST_PRICE_1', 'low_adj'),
          ('CLOSE_PRICE_1', 'close_adj'),
          ('TURNOVER_VOL', 'volume_adj')]),
        ('mkt_equd_eval',
         [('PE_T', 'pe_ttm'),
          ('PE_M', 'pe_mrq'),
          ('PE_CM', 'pe_deducted_mrq'),
          ('PB', 'pb_lf'),
          ('PS_T', 'ps_ttm'),
          ('PS_M', 'ps_mrq'),
          ('PCF_T', 'pcf_ncf_ttm'),
          ('PCF_M', 'pcf_ncf_mrq'),
          ('PCF_OT', 'pcf_ocf_ttm'),
          ('PCF_OM', 'pcf_ocf_mrq'),
          ('EV', 'ev'),
          ('EV_EBITDA', 'ev_2_ebitda'),
          ('EV_SALES', 'ev_2_sales_ttm')]),
        ('mkt_equd_ind',
         [('CHG_STATUS', 'trade_status'),
          ('CON_CHG', 'con_chg'),
          ('CON_LIMIT', 'con_limit'),
          ('DAYS_1', 'days_2_highest'),
          ('DAYS_2', 'days_new_highest'),
          ('DAYS_3', 'days_2_lowest'),
          ('DAYS_4', 'days_new_lowest'),
          ('DEAL_VALUE', 'deal_value')]),
        ('mkt_limit',
         [('LIMIT_UP_PRICE', 'limit_up'),
          ('LIMIT_DOWN_PRICE', 'limit_down'),
          ('UP_LIMIT_REACHED_TIMES', 'up_limit_reached_times'),
          ('DOWN_LIMIT_REACHED_TIMES', 'down_limit_reached_times')]),
        ('mkt_equ_vwapd', [('VWAP', 'vwap')])
    ]

    data_frames = []

    for table_name, column_mapping in table_column_mappings:
        df = select_hermes_columns_from_table(
            db=db,
            table_name=table_name,
            column_mapping=column_mapping,
            datayes_id_list=datayes_id_list,
            start_time=start_time,
            end_time=end_time,
        )

        exchange_map = {'XSHG': '.SH', 'XSHE': '.SZ'}
        df = df.assign(
            code=df['datayes_code'] + df['datayes_exchange_cd'].apply(
                lambda x: exchange_map.get(x, '')
            )
        )
        df.drop(
            columns=['datayes_id', 'datayes_code', 'datayes_exchange_cd'],
            inplace=True
        )
        df.rename({'trade_date': 'datetime'}, axis=1, inplace=True)
        df.set_index(['code', 'datetime'], inplace=True)
        data_frames.append(df)

    ret_df = pd.concat(data_frames, axis=1)
    ret_df.reset_index(inplace=True)
    ret_df = ret_df.assign(datetime=ret_df['datetime'].apply(
        lambda x: date(x.year, x.month, x.day)
    ))
    return ret_df


class App(ModuleApp):

    def process(self, queries: List[Query]):
        db = self.ctx.databases['hermes_data']

        for q in queries:

            getLogger(__name__).info('received query: %s', q)
            # if q.code_set.lower() not in ('', '*', 'all'):
            if q.code_sets != ['*']:
                raise RuntimeError('This module can only be operated on all codes.')

            df_industry = get_industry_info_from_hermes(db, '010317')

            # 最后写入增量
            self.s.write_table(
                str(TableNames.INDUSTRY),
                data=df_industry,
                layout=DataTableLayout.CODE_DATE,
                dtypes={'out_date': 'date'},
            )

            df = merge_hermes_daily_data(db, q.code_sets, q.start_time, q.end_time)

            #最后写入增量
            self.s.write_table(
                str(TableNames.DAILY),
                data=df,
                layout=DataTableLayout.CODE_DATE,
                dtypes={'out_date': 'date'},
            )

    def purge_cache(self, code_query: Optional[Query] = None):
        for table_name in TableNames.all_table_names():
            self.s.purge_table(
                table_name,
                codes=code_query.codes,
                code_sets=code_query.code_sets
            )


if __name__ == '__main__':
    run_module_app(App)
