from datetime import timedelta
from enum import Enum
from logging import getLogger
from typing import Optional, List

from quant_engine.core import *
from quant_engine.lib import *
from quant_engine.typing_ import *

from lib import daily_factors


class TableNames(str, Enum):
    DAILY = 'daily'

    def __str__(self):
        return self.value


DAILY_WINDOW_SIZE = timedelta(days=200)


class App(ModuleApp):

    def process(self, queries: List[Query]):
        for q in queries:
            getLogger(__name__).info('received query: %s', q)
            codes = self.s.read_codes(q.code_set)

            # daily
            daily_df = self.deps['common.prepare.daily_kdata'].s.read_table(
                'main',
                start_time=q.start_time - DAILY_WINDOW_SIZE,
                end_time=q.end_time,
                code_sets=[q.code_set],
                default_dtype=TableDType.DOUBLE,  # important: force decimal values casted into double
            )
            for code in codes:
                getLogger(__name__).info('compute daily factors, for code = %s, query = %s', code, q)
                code_df = mp_column_map(daily_factors, code=code, df=daily_df)
                self.s.write_table(
                    str(TableNames.DAILY),
                    data=code_df,
                    layout=DataTableLayout.CODE_DATE
                )

    def train_model(self, query: Query):
        pass

    def purge_cache(self, code_sets: Optional[List[str]] = None):
        for table_name in TableNames.__members__:
            self.s.purge_table(str(table_name), code_sets)


if __name__ == '__main__':
    run_module_app(App)
