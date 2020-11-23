import logging
from datetime import timedelta, date
from enum import Enum
from logging import getLogger

from quant_engine.core import *
from quant_engine.lib import *
from quant_engine.typing_ import *
from quant_engine.utils import *

from lib import *


class TableNames(BaseTableName):
    MAIN = 'main'


DAILY_WINDOW_SIZE = timedelta(days=365)


class App(ModuleApp):

    def process(self, queries: List[Query]):
        for q in queries:
            getLogger(__name__).info('received query: %s', q)
            codes = q.codes or self.s.read_codes(q.code_sets)

            for code in codes:
                getLogger(__name__).info('compute daily factors, for code = %s, query = %s', code, q)
                daily_df = self.deps['prepare.wash_daily'].s.read_table(
                    TableNames.MAIN,
                    start_time=q.start_time - DAILY_WINDOW_SIZE,
                    end_time=q.end_time,
                    codes=[code],
                    default_dtype=TableDType.DOUBLE,
                )

                # if len(daily_df) == 0:
                #     #未读到数据, 返回空集合
                #     continue

                result_df = mp_column_map(factors, daily_df)
                result_df.reset_index(inplace=True)
                result_df = result_df.assign(datetime=result_df['datetime'].apply(
                    lambda x: date(x.year, x.month, x.day)
                ))
                result_df = result_df[result_df['datetime'] >= q.start_time]

                self.s.write_table(
                    str(TableNames.MAIN),
                    data=result_df,
                    code=code,
                    layout=DataTableLayout.CODE_DATE
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
