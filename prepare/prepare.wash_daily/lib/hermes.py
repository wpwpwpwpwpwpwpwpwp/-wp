import pandas as pd
from typing import *
from quant_engine.core import *
from datetime import date, datetime


def select_hermes_columns_from_table(db: SqlDB,
                                     table_name: str,
                                     column_mapping: Sequence[Tuple[str, str]],
                                     datayes_id_list: List[str],
                                     start_time: Union[date, str],
                                     end_time: Union[date, str]):

    columns = [f't."{a}" AS "{b}"' for a, b in column_mapping]
    if not columns:
        raise ValueError(f'`column_mapping` must not be empty.')
    column_str = ',\n        '.join(columns)

    if datayes_id_list == ['*']:
        stmt = f'''
        SELECT
            t."TRADE_DATE" AS "trade_date",
            t2."SECURITY_ID" AS "datayes_id",
            t2."TICKER_SYMBOL" AS "datayes_code",
            t2."EXCHANGE_CD" AS "datayes_exchange_cd",
            {column_str}
        FROM
            "{table_name}" AS t
        LEFT JOIN
            "md_security" AS t2
        ON
            t."SECURITY_ID" = t2."SECURITY_ID"
        WHERE
            t."TRADE_DATE" >= %(start_time)s AND
            t."TRADE_DATE" < %(end_time)s
        ORDER BY
            t."TRADE_DATE" ASC,
            t."SECURITY_ID" ASC
        ;
        '''
    else:
        stmt = f'''
        SELECT
            t."TRADE_DATE" AS "trade_date",
            t2."SECURITY_ID" AS "datayes_id",
            t2."TICKER_SYMBOL" AS "datayes_code",
            t2."EXCHANGE_CD" AS "datayes_exchange_cd",
            {column_str}
        FROM
            "{table_name}" AS t
        LEFT JOIN
            "md_security" AS t2
        ON
            t."SECURITY_ID" = t2."SECURITY_ID"
        WHERE
            t."SECURITY_ID" = %(datayes_id)s AND
            t."TRADE_DATE" >= %(start_time)s AND
            t."TRADE_DATE" < %(end_time)s
        ORDER BY
            t."TRADE_DATE" ASC,
            t."SECURITY_ID" ASC
    ;
    '''
    with db.scoped_conn() as conn:
        if datayes_id_list != ['*']:
            df = pd.DataFrame()
            for datayes_id in datayes_id_list:
                df_each = pd.read_sql(
                        sql=stmt,
                        con=conn,
                        params={'datayes_id': datayes_id, 'start_time': start_time, 'end_time': end_time},
                )
                df = pd.concat([df, df_each], axis = 0)
        else:
            df = pd.read_sql(
                sql=stmt,
                con=conn,
                params={'start_time': start_time, 'end_time': end_time},
            )

        return df


def get_adj_info_from_hermes(db: SqlDB,
                             start_time: Union[date, str],
                             end_time: Union[date, str]):

    stmt = f'''
    SELECT
        t."EX_DIV_DATE" AS "ex_div_date",
        t."SECURITY_ID" AS "datayes_id",
        t."ADJ_FACTOR_1" AS "datayes_adjfactor",
        t."ACCUM_ADJ_FACTOR" AS "datayes_accum_adjfactor",
        t2."TICKER_SYMBOL" AS "datayes_code",
        t2."EXCHANGE_CD" AS "datayes_exchange_cd"
    FROM
        "mkt_adjf" as t
    LEFT JOIN
        "md_security" AS t2
    ON
        t."SECURITY_ID" = t2."SECURITY_ID"  
    WHERE
        t."EX_DIV_DATE" >= %(start_time)s AND
        t."EX_DIV_DATE" < %(end_time)s
    ORDER BY
        t."EX_DIV_DATE" DESC,
        t."TICKER_SYMBOL" ASC
    ;
    '''

    with db.scoped_conn() as conn:
        df = pd.read_sql(
            sql=stmt,
            con=conn,
            params={'start_time': start_time, 'end_time': end_time},
        )
        exchange_map = {'XSHG': '.SH', 'XSHE': '.SZ'}
        df = df.assign(
            code=df['datayes_code'] + df['datayes_exchange_cd'].apply(
                lambda x: exchange_map.get(x, '')
            ))
        df.loc[:,'date'] = df.ex_div_date
        return df


def get_adj_kdata_from_hermes(db: SqlDB,
                              datayes_id_list: List[str],
                              start_time: Union[date, str],
                              end_time: Union[date, str]):

    stmt = f'''
    SELECT
        t."PRE_CLOSE_PRICE_1" AS "pre_close_adj",
        t."OPEN_PRICE_1" AS "open_adj",
        t."HIGHEST_PRICE_1" AS "high_adj",
        t."LOWEST_PRICE_1" AS "low_adj",
        t."CLOSE_PRICE_1" AS "close_adj",
        t."TURNOVER_VOL" AS "volume_adj",
        t."TRADE_DATE" AS "date",
        t."TICKER_SYMBOL" AS "datayes_code",
        t."EXCHANGE_CD" AS "datayes_exchange_cd"
    FROM
        "mkt_equd_adj" as t
    WHERE   
        t."SECURITY_ID" = %(datayes_id)s AND
        t."TRADE_DATE" >= %(start_time)s AND
        t."TRADE_DATE" < %(end_time)s
    ORDER BY
        t."TRADE_DATE" DESC,
        t."TICKER_SYMBOL" ASC
    ;
    '''

    with db.scoped_conn() as conn:
        df = pd.DataFrame()
        for datayes_id in datayes_id_list:
            df_each = pd.read_sql(
                    sql=stmt,
                    con=conn,
                    params={'datayes_id': datayes_id, 'start_time': start_time, 'end_time': end_time},
            )
            df = pd.concat([df, df_each], axis = 0)

        exchange_map = {'XSHG': '.SH', 'XSHE': '.SZ'}
        df = df.assign(
             code=df['datayes_code'] + df['datayes_exchange_cd'].apply(lambda x: exchange_map.get(x, '')
             ))
        return df


def get_industry_info_from_hermes(db:SqlDB,
                                  industry_id:str):
    """

    :param db:
    :param industry_id:暂时只考虑提取单个行业，之后转为list
    :return:
    """
    stmt = f'''
    SELECT
        b."TICKER_SYMBOL" AS "datayes_code",
        b."SEC_SHORT_NAME" AS "sec_short_name",
        b."EXCHANGE_CD" AS "datayes_exchange_cd",
        c."TYPE_NAME" AS "type_name",
        c."TYPE_SYMBOL" AS "type_symbol",
        c."INDUSTRY" AS "industry",
        c."INDUSTRY_LEVEL" AS "industry_level",
        a."TYPE_ID" AS "datayes_type_id",
        a."INTO_DATE" AS "into_date",
        a."OUT_DATE" AS "out_date",
        a."IS_NEW" AS "is_new"

    FROM
        "md_inst_type" as a
        JOIN "md_security" as b ON a."PARTY_ID" = b."PARTY_ID"
        AND "ASSET_CLASS" = 'E'
        AND "EXCHANGE_CD" IN ('XSHG','XSHE')
        JOIN "md_type" as c ON a."TYPE_ID" = c."TYPE_ID"
        AND LEFT (c."TYPE_ID",6) = %(industry_id)s

    ORDER BY
        b."TICKER_SYMBOL" ASC,
        a."INTO_DATE" ASC
    ;
    '''

    with db.scoped_conn() as conn:

        df = pd.read_sql(sql=stmt,
                         con=conn,
                         params={'industry_id':industry_id})
        df.loc[:,'datetime'] = datetime.now().date()

        exchange_map = {'XSHG': '.SH', 'XSHE': '.SZ'}
        df = df.assign(code=df['datayes_code'] + df['datayes_exchange_cd'].apply(
                lambda x: exchange_map.get(x, '')))

        return df
