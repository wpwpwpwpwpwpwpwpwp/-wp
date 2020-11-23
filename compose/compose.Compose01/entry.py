import os
from logging import getLogger
from typing import *

import mltk
from quant_engine.core import *
from quant_engine.lib import *
from quant_engine.typing_ import *
from quant_engine.utils import *

from lib import *


class TableNames(BaseTableName):
    MAIN = 'main'


class App(ModuleApp):

    model: Optional[Model] = None

    def process(self, queries: List[Query]):
        # load the model
        if self.model is None:
            self.model = Model.load_file('./model/model.yml')

        # process the queries
        for q in queries:
            getLogger(__name__).info('process query: %s', q)
            df = load_feature_tables(
                app=self,
                patterns=[f.name for f in self.model.params.features],
                codes=q.codes,
                code_sets=q.code_sets,
                start_time=q.start_time,
                end_time=q.end_time,
            )
            print(df)
            # produce the (code, date, score) as result
            result_df = ...
            self.s.write_table(
                table_name=TableNames.MAIN,
                data=result_df,
                layout=DataTableLayout.CODE_DATE,
            )

    @app_entry('effective-test', use_args=['query'], use_unknown_args=True)
    def effective_test(self, query: Query):
        with mltk.Experiment(ModelConfig, output_dir='./', args=self.args) as exp:
            # load the features from table
            df = load_feature_tables(
                app=self,
                patterns=[
                    'feature.Bundle03/main/*',
                ],
                codes=query.codes,
                code_sets=query.code_sets,
                start_time=query.start_time,
                end_time=query.end_time,
            )

            # compute all necessary outputs and save to "./output/effective_test"

            # generate figures and reports into "./output/figures" and "./output/reports"

    @app_entry('compose', use_unknown_args=True)
    def compose(self):
        with mltk.Experiment(ModelConfig, output_dir='./', args=self.args) as exp:
            # use the config, e.g.,
            print(exp.config.compose.num_features)

            # read the output from "./output/effective_test"
            df = ...

            # select features according to config and output of effective test

            # now make the model
            model = Model(
                config=exp.config,
                params=ModelParams(
                    features=[
                        (feature, 1.0)
                        for feature in df
                    ]
                )
            )
            os.makedirs('./model', exist_ok=True)
            mltk.save_config(model, './model/model.yml', flatten=False)

    def purge_cache(self, code_query: Optional[Query] = None):
        for table_name in TableNames.all_table_names():
            self.s.purge_table(
                table_name,
                codes=code_query.codes,
                code_sets=code_query.code_sets
            )


if __name__ == '__main__':
    run_module_app(App)
