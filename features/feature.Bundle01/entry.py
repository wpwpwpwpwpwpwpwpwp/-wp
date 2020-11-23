import logging
from datetime import timedelta, date
from enum import Enum
from logging import getLogger
from typing import List, Optional

from quant_engine.core import *
from quant_engine.lib import *
from quant_engine.typing_ import *
from quant_engine.utils import *

from lib import *
from lib.utils import process_daily_df, process_minutely_df


class TableNames(BaseTableName):
    MAIN_MIN='min'
    MAIN_DAY='main_day'
    MAIN_FACTOR='smart_factor'


DAILY_WINDOW_SIZE = timedelta(days=60)
MINUTELY_WINDOW_SIZE = timedelta(days=60)


class App(ModuleApp):

    def process(self, queries: List[Query]):
        for q in queries:
            getLogger(__name__).info('received query: %s', q)
            codes = self.s.read_codes(q.code_set)

            for code in codes[0:100]:
                getLogger(__name__).info('compute minutely factors, for code = %s, query = %s', code, q)
                try:
                    minutely_df = self.deps['common.prepare.wash_wind'].s.read_table(
                        TableNames.MAIN_MIN,
                        start_time=str(pd.Timestamp(q.start_time)),
                        end_time=str(pd.Timestamp(q.end_time)),
                        codes=[code],
                        default_dtype=TableDType.DOUBLE,
                        ascending=True
                    )

                    daily_df = self.deps['common.prepare.wash_wind'].s.read_table(
                        TableNames.MAIN_DAY,
                        start_time=q.start_time,
                        end_time=q.end_time,
                        codes=[code],
                        default_dtype=TableDType.DOUBLE,
                        ascending=True
                    )
                except ValueError:
                    print('query stock:{} failed'.format(code))
                else:
                    if len(minutely_df) == 0 or len(daily_df) == 0:
                        #未读到数据, 返回空集合
                        continue

                    minutely_df = process_minutely_df(minutely_df)
                    daily_df = process_daily_df(daily_df)



                    result_df = mp_column_map(factors, code, daily_df, minutely_df)
                    result_df.reset_index(inplace=True)
                    result_df = result_df.assign(datetime=result_df['date'].apply(
                        lambda x: date(x.year, x.month, x.day)
                    ))
                    result_df = result_df[result_df['datetime'] >= q.start_time]

                    self.s.write_table(
                        str(TableNames.MAIN_FACTOR),
                        data=result_df,
                        layout=DataTableLayout.CODE_DATE
                    )

    def train_model(self, query: Query):

        pass

    def purge_cache(self, code_sets: Optional[List[str]] = None):
        for table_name in TableNames.__members__:
            self.s.purge_table(str(table_name), code_sets)


if __name__ == '__main__':
    run_module_app(App)
