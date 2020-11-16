import logging
from datetime import timedelta, date
from enum import Enum
from logging import getLogger

from quant_engine.core import *
from quant_engine.lib import *
from quant_engine.typing_ import *

from lib import *


class TableNames(str, Enum):
    MAIN = 'main'

    def __str__(self):
        return self.value


DAILY_WINDOW_SIZE = timedelta(days=365)


class App(ModuleApp):

    def process(self, queries: List[Query]):
        for q in queries:
            getLogger(__name__).info('received query: %s', q)
            codes = self.s.read_codes(q.code_set)

            for code in codes:
                getLogger(__name__).info('compute daily factors, for code = %s, query = %s', code, q)
                daily_df = self.deps['common.prepare.wash_daily'].s.read_table(
                    'main',
                    start_time=q.start_time - DAILY_WINDOW_SIZE,
                    end_time=q.end_time,
                    codes=[code],
                    # columns=[],
                    default_dtype=TableDType.DOUBLE,
                    ascending=True
                )
                daily_df.reset_index(inplace=True)

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

    def train_model(self, query: Query):
        pass

    def purge_cache(self, code_sets: Optional[List[str]] = None):
        for table_name in TableNames.__members__:
            self.s.purge_table(str(table_name), code_sets)


if __name__ == '__main__':
    run_module_app(App)
