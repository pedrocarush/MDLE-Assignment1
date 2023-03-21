import os.path
from pyspark import Broadcast
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import *
from pyspark.sql.types import StringType, ArrayType
from itertools import combinations
from typing import Iterable, Any, List, Tuple, Dict, Set

from argparse import ArgumentParser

"""
Python-file version of the Jupyter notebook, for submission via spark-submit.
The documentation is present in the notebook.
"""



def initialize_spark() -> SparkSession:
    return SparkSession.builder \
        .appName('SandboxAssign1') \
        .config('spark.master', 'local[*]') \
        .getOrCreate()


def prepare_data(spark: SparkSession, path: str) -> Tuple[Dict[str, str], DataFrame]:
    
    df = spark.read \
        .option('header', True) \
        .csv(path) \
        .drop('START', 'STOP', 'ENCOUNTER')

    code_description_map = {r.CODE: r.DESCRIPTION
        for r in df \
        .select('CODE', 'DESCRIPTION') \
        .distinct() \
        .collect()
    }

    df = df.drop('DESCRIPTION').distinct()

    return code_description_map, df


# A-priori Algorithm

def first_pass(
        spark: SparkSession,
        df: DataFrame,
        support_threshold: int,
        table_frequent_base: str
        ) -> Tuple[DataFrame, Broadcast]:
    
    fname = table_frequent_base + '_k1'

    if not os.path.exists(fname):
        frequent_diseases_k1 = df \
            .groupBy('CODE') \
            .count() \
            .withColumnRenamed('count', 'COUNT') \
            .filter(col('COUNT') >= support_threshold)
        
        frequent_diseases_k1.write.mode('overwrite').parquet(path=fname, compression='gzip')

    frequent_diseases_k1 = spark.read.parquet(fname)

    frequent_diseases_k1_set = {r.CODE for r in frequent_diseases_k1.select('CODE').collect()}
    frequent_diseases_k1_set = spark.sparkContext.broadcast(frequent_diseases_k1_set)

    return frequent_diseases_k1, frequent_diseases_k1_set


def second_pass(
        spark: SparkSession,
        df: DataFrame,
        frequent_diseases_k1_set: Broadcast,
        support_threshold: int,
        table_frequent_base: str
        ) -> Tuple[DataFrame, Set[str]]:

    @udf(returnType=ArrayType(ArrayType(StringType(), False), False))
    def combine_pairs(elems: Iterable[Any]):
        return list(combinations(elems, 2))

    fname = table_frequent_base + '_k2'

    if not os.path.exists(fname):
        frequent_diseases_k2 = df \
            .filter(col('CODE').isin(frequent_diseases_k1_set.value)) \
            .groupBy('PATIENT') \
            .agg(collect_list('CODE')) \
            .withColumn('collect_list(CODE)', array_sort('collect_list(CODE)')) \
            .withColumn('CODE_PAIRS', combine_pairs('collect_list(CODE)')) \
            .select('PATIENT', 'CODE_PAIRS') \
            .withColumn('CODE_PAIR', explode('CODE_PAIRS')) \
            .drop('CODE_PAIRS') \
            .groupBy('CODE_PAIR') \
            .count() \
            .withColumnRenamed('count', 'COUNT') \
            .filter(col('COUNT') >= support_threshold)
        
        frequent_diseases_k2.write.mode('overwrite').parquet(path=fname, compression='gzip')

    frequent_diseases_k2 = spark.read.parquet(fname)

    frequent_diseases_k2_set = {tuple(r.CODE_PAIR) for r in frequent_diseases_k2.select('CODE_PAIR').collect()}

    return frequent_diseases_k2, frequent_diseases_k2_set


def third_pass(
        spark: SparkSession,
        df: DataFrame,
        frequent_diseases_k1_set: Broadcast,
        frequent_diseases_k2_set: Set[str],
        support_threshold: int,
        table_frequent_base: str
        ) -> DataFrame:
    
    @udf(returnType=ArrayType(ArrayType(StringType(), False), False))
    def combine_triples(elems: Iterable[Any]):
        return [
            combination for combination in list(combinations(elems, 3))
            if ((combination[0], combination[1]) in frequent_diseases_k2_set
                and (combination[0], combination[2]) in frequent_diseases_k2_set
                and (combination[1], combination[2]) in frequent_diseases_k2_set)
        ]

    fname = table_frequent_base + '_k3'

    # TODO: is array_sort after the combination generation actually? watch out for the way spark partitions and stuff...?
    if not os.path.exists(fname):
        frequent_diseases_k3 = df \
            .filter(col('CODE').isin(frequent_diseases_k1_set.value)) \
            .groupBy('PATIENT') \
            .agg(collect_list('CODE')) \
            .withColumn('collect_list(CODE)', array_sort('collect_list(CODE)')) \
            .withColumn('CODE_TRIPLES', combine_triples('collect_list(CODE)')) \
            .select('PATIENT', 'CODE_TRIPLES') \
            .withColumn('CODE_TRIPLE', explode('CODE_TRIPLES')) \
            .drop('CODE_TRIPLES') \
            .groupBy('CODE_TRIPLE') \
            .count() \
            .withColumnRenamed('count', 'COUNT') \
            .filter(col('COUNT') >= support_threshold)

        frequent_diseases_k3.write.mode('overwrite').parquet(path=fname, compression='gzip')

    frequent_diseases_k3 = spark.read.parquet(fname)

    return frequent_diseases_k3


def write_most_frequent_diseases(
        code_description_map: Dict[str, str],
        frequent_diseases_k2: DataFrame,
        frequent_diseases_k3: DataFrame,
        most_frequent_base: str):
    
    @udf(returnType=ArrayType(StringType(), False))
    def map_codes_to_description(codes: List[str]):
        return [code_description_map[item] for item in codes]

    with open(most_frequent_base + '_k2.csv', 'w') as f:
        print('pair\tcount', file=f)
        print(*(
                f'{r.CODE_PAIR}\t{r.COUNT}' for r in
                frequent_diseases_k2
                    .withColumn('CODE_PAIR', map_codes_to_description('CODE_PAIR'))
                    .sort('COUNT', ascending=False).take(10)
            ), sep='\n', file=f)
        
    with open(most_frequent_base + '_k3.csv', 'w') as f:
        print('triple\tcount', file=f)
        print(*(
                f'{r.CODE_TRIPLE}\t{r.COUNT}' for r in
                frequent_diseases_k3
                    .withColumn('CODE_TRIPLE', map_codes_to_description('CODE_TRIPLE'))
                    .sort('COUNT', ascending=False).take(10)
            ), sep='\n', file=f)


def generate_association_rules(
        n_patients: int,
        frequent_diseases_k1: DataFrame,
        frequent_diseases_k2: DataFrame,
        frequent_diseases_k3: DataFrame,
        standardised_lift_threshold: float
        ) -> DataFrame:

    @udf(returnType=ArrayType(ArrayType(ArrayType(StringType(), False), False), False))
    def generate_association_rules(itemset: List[str]):
        itemset = set(itemset)
        return [(sorted(itemset - {item}), [item]) for item in itemset]


    rules_k2 = frequent_diseases_k2 \
        .withColumn('RULES', generate_association_rules('CODE_PAIR')) \
        .withColumn('RULES', explode('RULES')) \
        .select(col('RULES')[0].alias('RULE_LHS'), col('RULES')[1].alias('RULE_RHS'), col('COUNT').alias('COUNT_RULE')) \
        .join(frequent_diseases_k1, frequent_diseases_k1['CODE'] == col('RULE_LHS')[0], 'inner') \
        .withColumnRenamed('COUNT', 'COUNT_LHS') \
        .drop('CODE') \
        .join(frequent_diseases_k1, frequent_diseases_k1['CODE'] == col('RULE_RHS')[0], 'inner') \
        .withColumnRenamed('COUNT', 'COUNT_RHS') \
        .drop('CODE')

    def add_metrics_columns(rule_counts: DataFrame) -> DataFrame:
        return rule_counts \
            .withColumn('CONFIDENCE', col('COUNT_RULE') / col('COUNT_LHS')) \
            .withColumn('INTEREST', col('CONFIDENCE') - col('COUNT_RHS') / n_patients) \
            .withColumn('LIFT', n_patients * col('CONFIDENCE') / col('COUNT_RHS')) \
            .withColumn('STANDARDISED_LIFT', 
                        (col('LIFT') - array_max(array(
                            (col('COUNT_LHS') + col('COUNT_RHS')) / n_patients - 1,
                            lit(1 / n_patients)
                        )) / (col('COUNT_LHS') * col('COUNT_RHS') / (n_patients ** 2)))
                        /
                        ((n_patients / array_max(array(col('COUNT_LHS'), col('COUNT_RHS')))) - array_max(array(
                            (col('COUNT_LHS') + col('COUNT_RHS')) / n_patients - 1,
                            lit(1 / n_patients)
                        )) / (col('COUNT_LHS') * col('COUNT_RHS') / (n_patients ** 2)))
            )

    rules_k2_metrics = rules_k2 \
        .transform(add_metrics_columns) \
        .filter(col('STANDARDISED_LIFT') >= standardised_lift_threshold)

    rules_k3 = frequent_diseases_k3 \
        .withColumn('RULES', generate_association_rules('CODE_TRIPLE')) \
        .withColumn('RULES', explode('RULES')) \
        .select(col('RULES')[0].alias('RULE_LHS'), col('RULES')[1].alias('RULE_RHS'), col('COUNT').alias('COUNT_RULE')) \
        \
        .join(frequent_diseases_k1, array(frequent_diseases_k1['CODE']) == col('RULE_LHS'), 'left') \
        .withColumnRenamed('COUNT', 'COUNT_LHS') \
        .drop('CODE') \
        .join(frequent_diseases_k2, frequent_diseases_k2['CODE_PAIR'] == col('RULE_LHS'), 'left') \
        .withColumnRenamed('COUNT', 'COUNT_LHS_OTHER') \
        .drop('CODE_PAIR') \
        .withColumn('COUNT_LHS', coalesce('COUNT_LHS', 'COUNT_LHS_OTHER')) \
        .drop('COUNT_LHS_OTHER') \
        \
        .join(frequent_diseases_k1, array(frequent_diseases_k1['CODE']) == col('RULE_RHS'), 'inner') \
        .withColumnRenamed('COUNT', 'COUNT_RHS') \
        .drop('CODE')

    rules_k3_metrics = rules_k3 \
        .transform(add_metrics_columns) \
        .filter(col('STANDARDISED_LIFT') >= standardised_lift_threshold)

    assert rules_k2_metrics.columns == rules_k3_metrics.columns, 'The dataframes of rule metrics should have the same columns in the same order!'

    rules_metrics = rules_k2_metrics \
        .union(rules_k3_metrics) \
        .sort('STANDARDISED_LIFT', ascending=False)
    
    return rules_metrics


def write_association_rules(code_description_map: Dict[str, str], rules_metrics: DataFrame, association_rules_name: str):

    @udf(returnType=ArrayType(StringType(), False))
    def map_codes_to_description(codes: List[str]):
        return [code_description_map[item] for item in codes]

    @udf(returnType=StringType())
    def format_rule(rule_1: List[str], rule_2: List[str], *values: List[Any]):
        return f'{{{", ".join(rule_1)}}} -> {{{", ".join(rule_2)}}}: {", ".join(map(str, values))}'

    rules_metrics \
        .withColumn('RULE_LHS', map_codes_to_description('RULE_LHS')) \
        .withColumn('RULE_RHS', map_codes_to_description('RULE_RHS')) \
        .select(format_rule('RULE_LHS', 'RULE_RHS', 'STANDARDISED_LIFT', 'LIFT', 'CONFIDENCE', 'INTEREST').alias('LINE')) \
        .write.mode('overwrite').text(association_rules_name)



def main(
        dataset: str,
        support_threshold: int,
        standardised_lift_threshold: float,
        table_frequent_base: str,
        most_frequent_base: str,
        association_rules_name: str
):
    
    spark = initialize_spark()
    code_description_map, df = prepare_data(spark, dataset)

    frequent_diseases_k1, frequent_diseases_k1_set = first_pass(spark, df, support_threshold, table_frequent_base)
    frequent_diseases_k2, frequent_diseases_k2_set = second_pass(spark, df, frequent_diseases_k1_set, support_threshold, table_frequent_base)
    frequent_diseases_k3 = third_pass(spark, df, frequent_diseases_k1_set, frequent_diseases_k2_set, support_threshold, table_frequent_base)

    write_most_frequent_diseases(code_description_map, frequent_diseases_k2, frequent_diseases_k3, most_frequent_base)

    n_patients = df.select('PATIENT').distinct().count()

    rules_metrics = generate_association_rules(n_patients, frequent_diseases_k1, frequent_diseases_k2, frequent_diseases_k3, standardised_lift_threshold)

    write_association_rules(code_description_map, rules_metrics, association_rules_name)

    spark.stop()



if __name__ == '__main__':

    default_str = ' (default: %(default)s)'

    parser = ArgumentParser()
    parser.add_argument("file", type=str, help="path to the input dataset")
    parser.add_argument("-s", "--support", type=int, default=1000, help="support threshold" + default_str)
    parser.add_argument("-l", "--standardised-lift", type=float, default=0.2, help="standardised lift threshold" + default_str)
    parser.add_argument("-t", "--table-frequent-base", type=str, default="frequent_diseases", help="base name for the frequent itemsets parquets saved/loaded to/from disk" + default_str)
    parser.add_argument("-m", "--most-frequent-base", type=str, default="most_frequent", help="base name for the results on the most frequent itemsets saved to disk (WARNING: overwrites files named 'most_frequent_k{2,3}.csv')" + default_str)
    parser.add_argument("-a", "--association-rules-name", type=str, default="association_rules", help="name for the association rules saved to disk (WARNING: overwrites files with the same name)" + default_str)
    
    args = parser.parse_args()

    main(
        dataset=args.file,
        support_threshold=args.support,
        standardised_lift_threshold=args.standardised_lift,
        table_frequent_base=args.table_frequent_base,
        most_frequent_base=args.most_frequent_base,
        association_rules_name=args.association_rules_name
    )
