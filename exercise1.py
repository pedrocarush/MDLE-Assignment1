import os.path
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import StringType, ArrayType
from itertools import combinations, chain
from typing import Iterable, Any, List

from optparse import OptionParser

parser = OptionParser()
parser.add_option("-f", "--file", dest="file", help="input file")
parser.add_option("-s", "--support", dest="support", help="support threshold", default=1000)


(options, args) = parser.parse_args()

if options.file is None:
    print("Please specify the input file")
    exit(1)



# Spark Initialization
spark = SparkSession.builder \
    .appName('SandboxAssign1') \
    .config('spark.master', 'local[*]') \
    .getOrCreate()


# Prepare Data
df = spark.read \
    .option('header', True) \
    .csv('./data/conditions.csv.gz') \
    .drop('START', 'STOP', 'ENCOUNTER')


code_description_map = {r.CODE: r.DESCRIPTION
    for r in df \
    .select('CODE', 'DESCRIPTION') \
    .distinct() \
    .collect()
}


df = df.drop('DESCRIPTION').distinct()


# A-priori Algorithm


# First Pass

if not os.path.exists('frequent_diseases_k1'):
    frequent_diseases_k1 = df \
        .groupBy('CODE') \
        .count() \
        .withColumnRenamed('count', 'COUNT') \
        .filter(col('COUNT') >= options.support)
    
    frequent_diseases_k1.write.mode('overwrite').parquet(path='frequent_diseases_k1', compression='gzip')

frequent_diseases_k1 = spark.read.parquet('frequent_diseases_k1')


frequent_diseases_k1_set = {r.CODE for r in frequent_diseases_k1.select('CODE').collect()}


# Second Pass

@udf(returnType=ArrayType(ArrayType(StringType(), False), False))
def combine_pairs(elems: Iterable[Any]):
    return list(combinations(elems, 2))


if not os.path.exists('frequent_diseases_k2'):
    frequent_diseases_k2 = df \
        .filter(col('CODE').isin(frequent_diseases_k1_set)) \
        .groupBy('PATIENT') \
        .agg(collect_list('CODE')) \
        .withColumn('CODE_PAIRS', combine_pairs('collect_list(CODE)')) \
        .select('PATIENT', 'CODE_PAIRS') \
        .withColumn('CODE_PAIR', explode('CODE_PAIRS')) \
        .drop('CODE_PAIRS') \
        .withColumn('CODE_PAIR', array_sort('CODE_PAIR')) \
        .groupBy('CODE_PAIR') \
        .count() \
        .withColumnRenamed('count', 'COUNT') \
        .filter(col('COUNT') >= options.support)
    
    frequent_diseases_k2.write.mode('overwrite').parquet(path='frequent_diseases_k2', compression='gzip')

frequent_diseases_k2 = spark.read.parquet('frequent_diseases_k2')


frequent_diseases_k2_set = {tuple(r.CODE_PAIR) for r in frequent_diseases_k2.select('CODE_PAIR').collect()}


# Third Pass

@udf(returnType=ArrayType(ArrayType(StringType(), False), False))
def combine_triples(elems: Iterable[Any]):

    triples = list(combinations(elems, 3))

    triples = [
        combination for combination in triples
        if ((combination[0], combination[1]) in frequent_diseases_k2_set
            and (combination[0], combination[2]) in frequent_diseases_k2_set
            and (combination[1], combination[2]) in frequent_diseases_k2_set)
    ]
        
    return triples


if not os.path.exists('frequent_diseases_k3'):
    frequent_diseases_k3 = df \
        .filter(col('CODE').isin(frequent_diseases_k1_set)) \
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
        .filter(col('COUNT') >= options.support)

    frequent_diseases_k3.write.mode('overwrite').parquet(path='frequent_diseases_k3', compression='gzip')

frequent_diseases_k3 = spark.read.parquet('frequent_diseases_k3')


frequent_diseases_k3_set = {tuple(r.CODE_TRIPLE) for r in frequent_diseases_k3.select('CODE_TRIPLE').collect()}



# Most Frequent Diseases

with open('most_frequent_k2.txt', 'w') as f:
    print('pair\tcount', file=f)
    print(*(
            f'{r.CODE_PAIR}\t{r.COUNT}' for r in
            frequent_diseases_k2.sort('COUNT', ascending=False).take(10)
        ), sep='\n', file=f)

with open('most_frequent_k3.txt', 'w') as f:
    print('pair\tcount', file=f)
    print(*(
            f'{r.CODE_TRIPLE}\t{r.COUNT}' for r in
            frequent_diseases_k3.sort('COUNT', ascending=False).take(10)
        ), sep='\n', file=f)



# Association Rules

n_patients = df.select('PATIENT').distinct().count()


@udf(returnType=ArrayType(ArrayType(ArrayType(StringType(), False), False), False))
def inner_subsets(itemset: List[str]):
    itemset = set(itemset)
    combis = chain.from_iterable(list(map(set, combinations(itemset, k))) for k in range(1, len(itemset)))
    return list((sorted(combi), sorted(itemset - combi)) for combi in combis)


rules_k2 = frequent_diseases_k2 \
    .withColumn('CODE_PAIR', inner_subsets('CODE_PAIR')) \
    .withColumn('CODE_PAIR', explode('CODE_PAIR')) \
    .withColumnRenamed('COUNT', 'COUNT_PAIR') \
    .select(col('CODE_PAIR')[0].alias('RULE_1'), col('CODE_PAIR')[1].alias('RULE_2'), 'COUNT_PAIR') \
    .join(frequent_diseases_k1, frequent_diseases_k1['CODE'] == col('RULE_1')[0], 'inner') \
    .withColumnRenamed('COUNT', 'COUNT_1') \
    .drop('CODE') \
    .join(frequent_diseases_k1, frequent_diseases_k1['CODE'] == col('RULE_2')[0], 'inner') \
    .withColumnRenamed('COUNT', 'COUNT_2') \
    .drop('CODE')


rules_k2_metrics = rules_k2 \
    .withColumn('CONFIDENCE', col('COUNT_PAIR') / col('COUNT_1')) \
    .withColumn('INTEREST', col('CONFIDENCE') - col('COUNT_2') / n_patients) \
    .withColumn('LIFT', n_patients * col('CONFIDENCE') / col('COUNT_2')) \
    .withColumn('STANDARDISED_LIFT', 
                (col('LIFT') - array_max(array(
                    (col('COUNT_1') + col('COUNT_2')) / n_patients - 1,
                    lit(1 / n_patients)
                )) / (col('COUNT_1') * col('COUNT_2') / (n_patients ** 2)))
                /
                ((n_patients / array_max(array(col('COUNT_1'), col('COUNT_2')))) - array_max(array(
                    (col('COUNT_1') + col('COUNT_2')) / n_patients - 1,
                    lit(1 / n_patients)
                )) / (col('COUNT_1') * col('COUNT_2') / (n_patients ** 2)))
    ) \
    .filter(col('STANDARDISED_LIFT') >= 0.2) \
    .sort('STANDARDISED_LIFT', ascending=False)



rules_k3 = frequent_diseases_k3 \
    .withColumn('CODE_TRIPLE', inner_subsets('CODE_TRIPLE')) \
    .withColumn('CODE_TRIPLE', explode('CODE_TRIPLE')) \
    .withColumnRenamed('COUNT', 'COUNT_TRIPLE') \
    .select(col('CODE_TRIPLE')[0].alias('RULE_1'), col('CODE_TRIPLE')[1].alias('RULE_2'), 'COUNT_TRIPLE') \
    \
    .join(frequent_diseases_k1, array(frequent_diseases_k1['CODE']) == col('RULE_1'), 'left') \
    .withColumnRenamed('COUNT', 'COUNT_1') \
    .drop('CODE') \
    .join(frequent_diseases_k2, frequent_diseases_k2['CODE_PAIR'] == col('RULE_1'), 'left') \
    .withColumnRenamed('COUNT', 'COUNT_1_OTHER') \
    .drop('CODE_PAIR') \
    .withColumn('COUNT_1', coalesce('COUNT_1', 'COUNT_1_OTHER')) \
    .drop('COUNT_1_OTHER') \
    \
    .join(frequent_diseases_k1, array(frequent_diseases_k1['CODE']) == col('RULE_2'), 'left') \
    .withColumnRenamed('COUNT', 'COUNT_2') \
    .drop('CODE') \
    .join(frequent_diseases_k2, frequent_diseases_k2['CODE_PAIR'] == col('RULE_2'), 'left') \
    .withColumnRenamed('COUNT', 'COUNT_2_OTHER') \
    .drop('CODE_PAIR') \
    .withColumn('COUNT_2', coalesce('COUNT_2', 'COUNT_2_OTHER')) \
    .drop('COUNT_2_OTHER')


rules_k3_metrics = rules_k3 \
    .withColumn('CONFIDENCE', col('COUNT_TRIPLE') / col('COUNT_1')) \
    .withColumn('INTEREST', col('CONFIDENCE') - col('COUNT_2') / n_patients) \
    .withColumn('LIFT', n_patients * col('CONFIDENCE') / col('COUNT_2')) \
    .withColumn('STANDARDISED_LIFT', 
                (col('LIFT') - array_max(array(
                    (col('COUNT_1') + col('COUNT_2')) / n_patients - 1,
                    lit(1 / n_patients)
                )) / (col('COUNT_1') * col('COUNT_2') / (n_patients ** 2)))
                /
                ((n_patients / array_max(array(col('COUNT_1'), col('COUNT_2')))) - array_max(array(
                    (col('COUNT_1') + col('COUNT_2')) / n_patients - 1,
                    lit(1 / n_patients)
                )) / (col('COUNT_1') * col('COUNT_2') / (n_patients ** 2)))
    ) \
    .filter(col('STANDARDISED_LIFT') >= 0.2) \
    .sort('STANDARDISED_LIFT', ascending=False)


rules_k2_metrics.show(truncate=False)