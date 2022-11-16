#CLEC16AvF.py
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import scipy.stats as st
import seaborn as sns
version = '2022-11-11-'
'''
Key links:
EBI GWAS CATALOG:

RMI2
https://www.ebi.ac.uk/gwas/genes/RMI2
-> Studies, csv

CLEC16A
https://www.ebi.ac.uk/gwas/genes/CLEC16A
-> Studies, csv

CLEC16A OTG
https://genetics.opentargets.org/gene/ENSG00000038532
'''

#files
version = '2022-11-11-'
file1 = "EAvyse_HK_HS_imputed_2metal_6_24_2022.txt"
metal_wba = pd.read_csv(file1, lineterminator='\n', delim_whitespace=True)

metal_wba.columns.to_list()
metal_wba.columns = ['CHR:BP_hg19', 'CHR', 'SNP', 'POS', 'A1', 'A2', 'F_ABentham', 'F_UBentham', 'PBentham', 'ORBentham', '95%CIBentham', 'infoBentham', 'F_AWang', 'F_UWang', 'PWang', 'ORWang', '95%CIWang', 'infoWang', 'F_AAlarcon', 'F_UAlarcon', 'PAlarcon', 'ORAlarcon', '95%CIAlarcon', 'infoAlarcon', 'P_metal_SE', 'Direction', 'P_metal_SampleSize', 'HetPVal']

#make CHR:POS column with hg38 POS
metal_wba = metal_wba.assign(CP=lambda x: x['CHR'].astype('str') + ":" + x['POS'].astype('str'))
print("rows before balloon filtering: "+ str(len(metal_wba.index)))
#11963275

#filter out the bentham balloons
balloons1 = ["rs7285053", "rs8078864", "rs17091347", "rs73050535", "rs1034009", "rs10264693", "rs17168663", "rs16895550", "rs9969061", "rs11962557", "rs13170409", "rs6532924", "rs17087866", "rs12081621", "rs4074976", "rs4661543", "rs13019891", "rs512681", "rs10200680", "rs2573219","rs9852014", "rs1464446", "rs1078324", "rs7386188", "rs7823055", "rs11928304", "rs12309414", "rs12948819"]
balloons2 = ["rs34703115","rs55684314","rs77601723","rs73068668","20:7357645:i","rs111478576","rs1120809","rs11905662","rs11907821","rs2423194","rs34995148","rs56412650","rs57830939","rs58072943","rs58605847","rs58829623","rs60461307","rs7262084","rs73894580","rs73894583","rs73894585","rs73894599","rs73897155","rs77222522"]
balloons = balloons1+balloons2
print("# of balloons: " + str(len(balloons)))
balloonmask = metal_wba.SNP.isin(balloons)
metal_wba = metal_wba[~balloonmask]
print("rows after balloon filtering: "+ str(len(metal_wba.index)))
#11963223

''' METAL - SNP duplicates '''
firstiddups = metal_wba.duplicated(subset=['SNP'], keep='first')
lastiddups = metal_wba.duplicated(subset=['SNP'], keep='last')
print("duplicated ids in metal: ")
print(metal_wba[firstiddups])
print(metal_wba[lastiddups])
metal_wba = metal_wba.drop_duplicates(subset=['SNP'], keep='last')
print("rows after duplicate SNP filtering: "+ str(len(metal_wba.index)))
#11963221

''' METAL – CP duplicates '''
# spacer
firstcpdups = metal_wba.duplicated(subset=['CP'], keep='first')
lastcpdups = metal_wba.duplicated(subset=['CP'], keep='last')
print("duplicated CP in metal: ")
print(metal_wba[firstcpdups])
print(metal_wba[lastcpdups])
metal_wba = metal_wba.drop_duplicates(subset=['CP'], keep='last')
print("rows after duplicate CP filtering: "+ str(len(metal_wba.index)))
#11963214

''' Generate betas, SE '''
'''
vectorized versions as below with assign are blindingly fast
when compared with .apply
'''

'''Calculate betas '''
metal_wba = metal_wba.assign(BetaBentham=lambda x:np.log(x['ORBentham']))
metal_wba = metal_wba.assign(BetaWang=lambda x:np.log(x['ORWang']))
metal_wba = metal_wba.assign(BetaAlarcon=lambda x:np.log(x['ORAlarcon']))
'''Calculate LCI & UCI'''
''' nans are ok with vectorised version '''

metal_wba[['LCIBentham', 'UCIBentham']] = metal_wba['95%CIBentham'].str.split("-", 1, expand=True)
metal_wba[['LCIWang', 'UCIWang']] = metal_wba['95%CIWang'].str.split("-", 1, expand=True)
metal_wba[['LCIAlarcon', 'UCIAlarcon']] = metal_wba['95%CIAlarcon'].str.split("-", 1, expand=True)

'''type cast as float64'''
metal_wba['LCIBentham'] = metal_wba['LCIBentham'].astype('float64')
metal_wba['UCIBentham'] = metal_wba['UCIBentham'].astype('float64')
metal_wba['LCIWang'] = metal_wba['LCIWang'].astype('float64')
metal_wba['UCIWang'] = metal_wba['UCIWang'].astype('float64')
metal_wba['LCIAlarcon'] = metal_wba['LCIAlarcon'].astype('float64')
metal_wba['UCIAlarcon'] = metal_wba['UCIAlarcon'].astype('float64')

'''Calculate log(OR) SE aka SE'''
#np.log(OR) - np.log(LCI) / 1.96
metal_wba = metal_wba.assign(SEBentham=lambda x: (np.log(x['ORBentham'])-np.log(x['LCIBentham']))/1.96)
metal_wba = metal_wba.assign(SEWang=lambda x: (np.log(x['ORWang'])-np.log(x['LCIWang']))/1.96)
metal_wba = metal_wba.assign(SEAlarcon=lambda x: (np.log(x['ORAlarcon'])-np.log(x['LCIAlarcon']))/1.96)

sum(metal_wba.SEBentham.gt(0))
sum(metal_wba.SEWang.gt(0))
sum(metal_wba.SEAlarcon.gt(0))

sum(metal_wba.SEBentham.lt(0))
sum(metal_wba.SEWang.lt(0))
sum(metal_wba.SEAlarcon.lt(0))

print(metal_wba.columns.to_list())
metal_columns = ['CHR:BP_hg19', 'CHR', 'SNP', 'POS', 'A1', 'A2', 'F_ABentham', 'F_UBentham', 'PBentham', 'ORBentham', '95%CIBentham', 'infoBentham', 'F_AWang', 'F_UWang', 'PWang', 'ORWang', '95%CIWang', 'infoWang', 'F_AAlarcon', 'F_UAlarcon', 'PAlarcon', 'ORAlarcon', '95%CIAlarcon', 'infoAlarcon', 'P_metal_SE', 'Direction', 'P_metal_SampleSize', 'HetPVal', 'CP', 'BetaBentham', 'BetaWang', 'BetaAlarcon', 'LCIBentham', 'UCIBentham', 'LCIWang', 'UCIWang', 'LCIAlarcon', 'UCIAlarcon', 'SEBentham', 'SEWang', 'SEAlarcon']

common_columns = ['CP',  'A1', 'A2', 'SNP', 'CHR:BP_hg19', 'CHR', 'POS']

bentham_filter = filter(lambda a: 'Bentham' in a, metal_columns)
wang_filter = filter(lambda a: 'Wang' in a, metal_columns)
alarcon_filter = filter(lambda a: 'Alarcon' in a, metal_columns)

''' now subset METAL files '''
benthamcolumns = common_columns + list(bentham_filter)
wangcolumns = common_columns + list(wang_filter)
alarconcolumns = common_columns + list(alarcon_filter)

''' make copies '''
benthamgwas = metal_wba[benthamcolumns].copy()
wanggwas = metal_wba[wangcolumns].copy()
alarcongwas = metal_wba[alarconcolumns].copy()

values = {"infoWang":1}
metal_wba.fillna(values, inplace=True)

'''Drop the empty rows'''
benthamgwas.dropna(axis=0, subset=['PBentham', 'ORBentham'], inplace=True)
wanggwas.dropna(axis=0, subset=['PWang', 'ORWang'], inplace=True)
alarcongwas.dropna(axis=0, subset=['PAlarcon', 'ORAlarcon'], inplace=True)
print("rows from bentham : "+ str(len(benthamgwas.index)))
print("rows from wang : "+ str(len(wanggwas.index)))
print("rows from alarcon(Fizi Imputed) : "+ str(len(alarcongwas.index)))

''' lists to select the output columns – oc '''
oc_bentham = ['CP', 'A1', 'A2', 'SNP',  'PBentham', 'BetaBentham', 'SEBentham']
oc_wang = ['CP', 'A1', 'A2', 'SNP', 'PWang', 'BetaWang', 'SEWang']
oc_alarcon = ['CP', 'A1', 'A2','SNP', 'PAlarcon', 'BetaAlarcon', 'SEAlarcon']

''' Save the individual harmonized gwas files '''
benthamgwas[oc_bentham].to_csv('bentham_metal_format'+version+'.txt', sep='\t', index=False)
wanggwas[oc_wang].to_csv('wang_metal_format'+version+'.txt', sep='\t', index=False)
alarcongwas[oc_alarcon].to_csv('alarcon_metal_format'+version+'.txt', sep='\t', index=False)

''' output full file here too, since we now have a cp column in the METAL file '''
metal_wba.to_csv('EAvyse_HK_HS_imputed_2metal_itwh'+version+'.txt', sep='\t', index=False)



'''
several additional balloons noted in Benthm with sample size meta-analysis
rs34703115
rs55684314
rs77601723
rs73068668
20:7357645:i
rs111478576
rs1120809
rs11905662
rs11907821
rs2423194
rs34995148
rs56412650
rs57830939
rs58072943
rs58605847
rs58829623
rs60461307
rs7262084
rs73894580
rs73894583
rs73894585
rs73894599
rs73897155
rs77222522
All are only supported by bentham study and are relatively rare or specific to european ancestry
'''

balloons2 = ["rs34703115","rs55684314","rs77601723","rs73068668","20:7357645:i","rs111478576","rs1120809","rs11905662","rs11907821","rs2423194","rs34995148","rs56412650","rs57830939","rs58072943","rs58605847","rs58829623","rs60461307","rs7262084","rs73894580","rs73894583","rs73894585","rs73894599","rs73897155","rs77222522"]

mask_additional_balloons = metal_wba.SNP.isin(balloons2)
metal_wba.drop(metal_wba[mask_additional_balloons].index, inplace=True)
sum(mask_additional_balloons)

metal_wba.columns.to_list()
'''
['CHR:BP_hg19', 'CHR', 'SNP', 'POS', 'A1', 'A2', 'F_ABentham', 'F_UBentham', 'PBentham', 'ORBentham', '95%CIBentham', 'infoBentham', 'F_AWang', 'F_UWang', 'PWang', 'ORWang', '95%CIWang', 'infoWang', 'F_AAlarcon', 'F_UAlarcon', 'PAlarcon', 'ORAlarcon', '95%CIAlarcon', 'infoAlarcon', 'P_metal_SE', 'Direction', 'P_metal_SampleSize', 'HetPVal', 'CP', 'BetaBentham', 'BetaWang', 'BetaAlarcon', 'LCIBentham', 'UCIBentham', 'LCIWang', 'UCIWang', 'LCIAlarcon', 'UCIAlarcon', 'SEBentham', 'SEWang', 'SEAlarcon']
'''

metal_wba = metal_wba.assign(CPA2A1=lambda x: x['CP'].astype('str') + "_" + x['A2'].astype('str') + "/" + x['A1'].astype('str'))
sum(metal_wba.CPA2A1.isna())

hoc_bentham = ['CPA2A1', 'A1', 'A2', 'SNP',  'PBentham', 'BetaBentham', 'SEBentham',  'F_UBentham']
hoc_wang = ['CPA2A1','A1', 'A2', 'SNP', 'PWang', 'BetaWang', 'SEWang', 'F_UWang']
hoc_alarcon = ['CPA2A1','A1', 'A2', 'SNP', 'PAlarcon', 'BetaAlarcon', 'SEAlarcon', 'F_UAlarcon']

hoc_bentham_rows = ~(metal_wba.PBentham.isna() | metal_wba.BetaBentham.isna() | metal_wba.SEBentham.isna())
hoc_wang_rows = ~(metal_wba.PWang.isna() | metal_wba.BetaWang.isna() | metal_wba.SEWang.isna())
hoc_alarcon_rows = ~(metal_wba.PAlarcon.isna() | metal_wba.BetaAlarcon.isna()| metal_wba.SEAlarcon.isna())

sum(hoc_bentham_rows)
10158694
sum(hoc_wang_rows)
5111726
sum(hoc_alarcon_rows)
6823438

metal_wba[hoc_bentham][hoc_bentham_rows].to_csv('bentham_metal_format'+version+'.txt', sep="\t", index=False)
metal_wba[hoc_wang][hoc_wang_rows].to_csv('wang_metal_format'+version+'.txt', sep="\t", index=False)
metal_wba[hoc_alarcon][hoc_alarcon_rows].to_csv('alarcon_metal_format'+version+'.txt', sep="\t", index=False)



'''*****************************************************************************
4) run meta-analyses:
Sumstats should be:
[X] 1. bentham
[X] 2. wang
[X] 3. alarcon
[X] 8. bentham + wang + alarcon

Define all regions in these sumstats
pull all of these index markers/regions from the METAL + AAGWAS common joined file
*****************************************************************************'''

'''
Setup the metalscript files ...
'''

scheme_samplesize = """
#This is sample size weighted std error meta-analysis uses p-value and direction of effect, weighted according to sample size
SCHEME SAMPLESIZE
"""

scheme_stderr = """
#This is inverse variance weighted std error meta-analysis - classical approach, uses effect size estimates and standard errors
SCHEME STDERR
"""

bentham_metal_header = """
#Bentham et al. SLE GWAS
MARKER CPA2A1
ALLELE A1 A2
EFFECT BetaBentham
STDERRLABEL SEBentham
PVAL PBentham
DEFAULTWEIGHT 10995

"""
bentham_metal_process = "PROCESS ./bentham_metal_format"+version+".txt\n"

wang_metal_header = """
#Wang et al. SLE GWAS
MARKER CPA2A1
ALLELE A1 A2
EFFECT BetaWang
STDERRLABEL SEWang
PVAL PWang
DEFAULTWEIGHT 12653

"""
wang_metal_process = "PROCESS ./wang_metal_format"+version+".txt\n"

alarcon_metal_header = """
#Alarcon et al. SLE GWAS
MARKER CPA2A1
ALLELE A1 A2
EFFECT BetaAlarcon
STDERRLABEL SEAlarcon
PVAL PAlarcon
DEFAULTWEIGHT 2279

"""
alarcon_metal_process = "PROCESS ./alarcon_metal_format"+version+".txt\n"

#Total n 37026 = 7500 + ~29526
analyze_quit = """
ANALYZE HETEROGENEITY
QUIT
"""

'''
********************************************************************************
METAL 2 : Bentham, Wang, Alarcon
********************************************************************************
'''

metal2_header = """
#METAL #2 – Bentham, Wang, Alarcon
#EAvyse_HK_HS_imputed_2metal_6_24_2022.txt +
#AA SLE GWAS : Sample size Weighted Meta-analysis
#EAvyse_HK_HS_imputed_2metal_6_24_2022.txt processed into individual files
#by metal_bwak_variant_lineup_v5-2022-06-25.py
"""

metalfile2 = "bentham-wang-alarcon"+version+".txt"
outfile2 = "OUTFILE META_Bentham_Wang_Alarcon_SampleSize_CPA2A1 .tbl\n"

metalscript2 = metal2_header + scheme_samplesize + \
bentham_metal_header + bentham_metal_process + \
wang_metal_header + wang_metal_process + \
alarcon_metal_header + alarcon_metal_process + \
outfile2 + analyze_quit

with open(metalfile2, 'w') as f:
    f.write(metalscript2)
    f.close()

metal = "./metal"
os.system(metal + " < " + metalfile2)

'''
set up output columns
'''
oc_metal =['MarkerName', 'P']
oc_gwas = ['CPA2A1', 'P', 'Beta', 'SE', 'F_U']

'''
********************************************************************************
 1 - bentham gwas
********************************************************************************
'''
#base refers to basename of the file. excluding .txt and will add when we open
base1 = "bentham_metal_format"+version

#read in files
bentham_gwas = pd.read_csv(base1+".txt", lineterminator='\n', delim_whitespace=True)

'''pre-process bentham GWAS '''
bentham_gwas[['CHR', 'POS']] = bentham_gwas['CPA2A1'].str.split(r":|_", 2, expand=True)[[0,1]]
bentham_gwas.columns = ['CPA2A1', 'A1', 'A2', 'ID', 'P', 'Beta', 'SE', 'F_U', 'CHR', 'POS']
bentham_gwas = bentham_gwas.astype({'CHR' : 'int32', 'POS' : 'int32'})

bentham_gwas = bentham_gwas.sort_values(by=['CHR', 'POS'])
bentham_gwas.equals(bentham_gwas.sort_values(by=['CHR', 'POS']))
bentham_gwas[oc_gwas].to_csv('bentham_gwas_lz'+version+'.txt', sep='\t', index=False)
os.system("gzip -k " + 'bentham_gwas_lz'+version+'.txt &')

'''
********************************************************************************
 2 - wang gwas
********************************************************************************
'''
base2 = "wang_metal_format"+version
wang_gwas = pd.read_csv(base2 +".txt", lineterminator='\n', delim_whitespace=True)
wang_gwas[['CHR', 'POS']] = wang_gwas['CPA2A1'].str.split(r":|_", 2, expand=True)[[0,1]]

wang_gwas.columns = ['CPA2A1', 'A1', 'A2', 'ID', 'P', 'Beta', 'SE', 'F_U', 'CHR', 'POS']
wang_gwas = wang_gwas.astype({'CHR' : 'int32', 'POS' : 'int32'})

wang_gwas = wang_gwas.sort_values(by=['CHR', 'POS'])
wang_gwas.equals(wang_gwas.sort_values(by=['CHR', 'POS']))
wang_gwas[oc_gwas].to_csv('wang_gwas_lz'+version+'.txt', sep='\t', index=False)
os.system("gzip -k " + 'wang_gwas_lz'+version+'.txt &')

'''
********************************************************************************
 3 - alarcon gwas
********************************************************************************
'''
base3 = "alarcon_metal_format"+version
alarcon_gwas = pd.read_csv(base3 +".txt", lineterminator='\n', delim_whitespace=True)
alarcon_gwas[['CHR', 'POS']] = alarcon_gwas['CPA2A1'].str.split(r":|_", 2, expand=True)[[0,1]]

alarcon_gwas.columns = ['CPA2A1', 'A1', 'A2', 'ID', 'P', 'Beta', 'SE', 'F_U', 'CHR', 'POS']
alarcon_gwas = alarcon_gwas.astype({'CHR' : 'int32', 'POS' : 'int32'})

alarcon_gwas = alarcon_gwas.sort_values(by=['CHR', 'POS'])
alarcon_gwas.equals(alarcon_gwas.sort_values(by=['CHR', 'POS']))
alarcon_gwas[oc_gwas].to_csv('alarcon_gwas_lz'+version+'.txt', sep='\t', index=False)
os.system("gzip -k " + 'alarcon_gwas_lz'+version+'.txt &')

oc_alarcon = ['CHR', 'POS','A2', 'A1', 'P']
alarcon_gwas = alarcon_gwas.sort_values(by=['CHR', 'POS'])
alarcon_gwas[oc_alarcon].to_csv('alarcon_gwas_lzponly'+version+'.txt', sep='\t', index=False)
os.system("gzip -k " + 'alarcon_gwas_lzponly'+version+'.txt &')

'''
********************************************************************************
 4 - wba gwas metal
********************************************************************************
'''
''' 6– META - bentham wang alarcon '''
oc_metal =['MarkerName', 'P']
oc_gwas = ['CPA2A1', 'P', 'Beta', 'SE', 'F_U']

base6 = "META_Bentham_Wang_Alarcon_SampleSize_CPA2A11"
metal_bwa = pd.read_csv(base6 +".tbl", lineterminator='\n', delim_whitespace=True)

metal_bwa[['CHR', 'POS']] = metal_bwa['MarkerName'].str.split(r":|_", 2, expand=True)[[0,1]]

metal_bwa.columns = ['MarkerName', 'Allele1', 'Allele2', 'Weight', 'Zscore', 'P', 'Direction', 'HetISq', 'HetChiSq', 'HetDf', 'HetPVal', 'CHR', 'POS']
metal_bwa = metal_bwa.astype({'CHR' : 'int32', 'POS' : 'int32'})
metal_bwa = metal_bwa.sort_values(by=['CHR', 'POS'])

metal_bwa.equals(metal_bwa.sort_values(by=['CHR', 'POS']))

metal_bwa[oc_metal].to_csv('metal_wba_lz'+version+'.txt', sep='\t', index=False)
os.system("gzip -k " + 'metal_wba_lz'+version+'.txt &')

'''
Problem with Bentham study plotting, so try again with P only
'''
oc_bentham = ['CHR', 'POS','A2', 'A1', 'P']
bentham_gwas = bentham_gwas.sort_values(by=['CHR', 'POS'])
bentham_gwas.equals(bentham_gwas.sort_values(by=['CHR', 'POS']))
bentham_gwas[oc_bentham].to_csv('bentham_gwas_lzponly'+version+'.txt', sep='\t', index=False)
os.system("gzip -k " + 'bentham_gwas_lzponly'+version+'.txt &')

'''
Triangles (effect size/direction) will switch to circles (P-value only)
'''
oc_wang = ['CHR', 'POS','A2', 'A1', 'P']
wang_gwas = wang_gwas.sort_values(by=['CHR', 'POS'])
wang_gwas.equals(wang_gwas.sort_values(by=['CHR', 'POS']))
wang_gwas[oc_wang].to_csv('wang_gwas_lzponly'+version+'.txt', sep='\t', index=False)
os.system("gzip -k " + 'wang_gwas_lzponly'+version+'.txt &')

'''
IMPORTANT
Window for viewing as defined by SLE gwas metal signal is:
16:10846031-11346031
'''

'''
********************************************************************************
********************************************************************************
cross-disease comparison
********************************************************************************
********************************************************************************
'''

'''
********************************************************************************
139 Studies with association in this region.
Criteria:
has summary statistic
Immune mediated
no multiples of the same or closely related phenotype (clarity of relationship)
– i.e. do not want atopy, asthma, eczema, eosinophils X 3 each
hg38 harmonized summmary statistics

This is the format (see readme in harmonised sum stats per EBI GWAS catalog)
--------------------------------
Summary Statistics Harmonisation
--------------------------------

Within this directory, <author>_<pmid>_<study_accession>/, in addition to the
author's files, there is the following:

harmonised/
 |- formmatted_file
 |- harmonised_file

'formatted_file' adheres to pattern:
    <pmid>-<study_accession>-<EFO_trait>-<build>.f.tsv.gz

'harmonised_file' adheres to pattern:
    <pmid>-<study_accession>-<EFO_trait>.h.tsv.gz


--------------------------------------------------------------------------------

Why does this exist?

At the time of writing, there is no standard to which summary statistics file
are made to. We are formatting and harmonising author contributed summary
statistic files to enable users to access data with ease.

For summary statistics FAQ please see: https://www.ebi.ac.uk/gwas/docs/faq

For more on summary statistics methods please see: https://www.ebi.ac.uk/gwas/docs/methods/summary-statistics

--------------------------------------------------------------------------------

What are the formatted and harmonised files?

Formatted files:
The formatted file is the summary statistics file after it has undergone a
semi-automated formatting process to bring it to a 'common' format. The goal is
to enable users to access all the raw data given in the author's summary
statistics file, but with the headers convertered to a consistent format seen
across all formatted summary statistics files.

- Headers will be coereced to the 'common format'.
- Rows will never be removed.
- Columns may be split, merged, deleted, added or moved.
- Values will be unaltered.
- Blanks will be set to 'NA'

Formatted file headings (not all may be present in file):

    'variant_id' = variant ID
    'p-value' = p-value
    'chromosome' = chromosome
    'base_pair_location' = base pair location
    'odds_ratio' = odds ratio
    'ci_lower' = lower 95% confidence interval
    'ci_upper' = upper 95% confidence interval
    'beta' = beta
    'standard_error' = standard error
    'effect_allele' = effect allele
    'other_allele' = other allele
    'effect_allele_frequency' = effect allele frequency

Note that the headers in the formatted file are not limited to the above
headers, nor are they required to have all of them.

Harmonised files:

The goal of the harmonisation is to make available the summary
statistics in a uniform, harmonised way. That means:

1) All location data will be on the same and genome build (GRCh38)

2) Missing data that can be inferred will, if safely possible.

3) All data will be harmonised to the same orientation.

The harmonised file is the output of formatted file after is has undergone an
automated harmonisation process (repo:
https://github.com/EBISPOT/sum-stats-formatter/tree/master/harmonisation).

  Mapping variant IDs to locations
   1) Update base pair location value by mapping variant ID using
      Ensembl release 93
      OR
      if above not possible, liftover base pair location to latest genome build
      OR
      if above not possible, remove variant from file.

  Harmonisation (repo: https://github.com/opentargets/sumstat_harmoniser)
   2) Using chromosome, base pair location and the effect and other alleles,
      check the orientation of all non-palindromic variants against Ensembl VCF
      references to detemine consensus:
      --> forward
      --> reverse
      --> mixed
      If the consensus is 'forward' or 'reverse', the following harmonisation
      steps are performed on the palindromic variants, with the assumption that
      they are orientated according to the consensus, otherwise palindromic
      variants are not harmonised.
   3) Using chromosome, base pair location and the effect and other alleles,
      query each variant against the Ensembl release 93 VCF reference to harmonise as
      appropriate by either:
      --> keeping record as is because:
          - it is already harmonised
          - it cannot be harmonised
      --> orientating to reference strand:
          - reverse complement the effect and other alleles
      --> flipping the effect and other alleles
          - because the effect and other alleles are flipped in the reference
          - this also means the beta, odds ratio, 95% CI and effect allele
            frequency are inverted
      --> a combination of the orientating and flipping the alleles.
      The result of the harmonisation is the addition of a set of new fields
      for each record (see below). A harmonisation code is assigned to each
      record indicating the harmonisation process that was performed (note
      that at the time of writing anyi processes involving 'Infer strand' are
      not being used):

     +----+--------------------------------------------------------------+
     |Code|Description of harmonisation process                          |
     +----+--------------------------------------------------------------+
     |1   |Palindromic; Infer strand; Forward strand; Alleles correct    |
     |2   |Palindromic; Infer strand; Forward strand; Flipped alleles    |
     |3   |Palindromic; Infer strand; Reverse strand; Alleles correct    |
     |4   |Palindromic; Infer strand; Reverse strand; Flipped alleles    |
     |5   |Palindromic; Assume forward strand; Alleles correct           |
     |6   |Palindromic; Assume forward strand; Flipped alleles           |
     |7   |Palindromic; Assume reverse strand; Alleles correct           |
     |8   |Palindromic; Assume reverse strand; Flipped alleles           |
     |9   |Palindromic; Drop palindromic; Not harmonised                 |
     |10  |Forward strand; Alleles correct                               |
     |11  |Forward strand; Flipped alleles                               |
     |12  |Reverse strand; Alleles correct                               |
     |13  |Reverse strand; Flipped alleles                               |
     |14  |Required fields are not known; Not harmonised                 |
     |15  |No matching variants in reference VCF; Not harmonised         |
     |16  |Multiple matching variants in reference VCF; Not harmonised   |
     |17  |Palindromic; Infer strand; EAF or reference VCF AF not known; |
     |    |Not harmonised                                                |
     |18  |Palindromic; Infer strand; EAF < specified minor allele       |
     |    |frequency threshold; Not harmonised                           |
     +----+--------------------------------------------------------------+

  Filtering and QC
    4) Variant ID is set to variant IDs found by step (5).
    5) Records without a valid value for variant ID, chromosome, base pair
       location and p-value are removed.

- Headers will be coerced to the 'harmonised format'.
- Addition harmonised data columns will be added.
- Rows may be removed.
- Variant ID, chromosome and base pair location may change (likely).


Harmonised file headings (not all may be present in file):

    'variant_id' = variant ID
    'p-value' = p-value
    'chromosome' = chromosome
    'base_pair_location' = base pair location
    'odds_ratio' = odds ratio
    'ci_lower' = lower 95% confidence interval
    'ci_upper' = upper 95% confidence interval
    'beta' = beta
    'standard_error' = standard error
    'effect_allele' = effect allele
    'other_allele' = other allele
    'effect_allele_frequency' = effect allele frequency
    'hm_variant_id' = harmonised variant ID
    'hm_odds_ratio' = harmonised odds ratio
    'hm_ci_lower' = harmonised lower 95% confidence interval
    'hm_ci_upper' =  harmonised lower 95% confidence interval
    'hm_beta' = harmonised beta
    'hm_effect_allele' = harmonised effect allele
    'hm_other_allele' = harmonised other allele
    'hm_effect_allele_frequency' = harmonised effect allele frequency
    'hm_code = harmonisation code (to lookup in 'Harmonisation Code Table')

********************************************************************************
'''

'''
Study ID – GWAS/IC – GWAS # – ancestry
Cordell PBC – GWAS – GCST003129 – European
Beecham MS – ImmunoChip – GCST005531 – European
Liu PBC – ImmunoChip – GCST005581 – European
Sawcer MS – GWAS – GCST001198 – European
Onengut T1D – ImmunoChip – GCST005536 – European
bentham sle – GWAS – GCST003156 – European
Ji psc – GWAS – GCST004030 – European
Chen a1c – GWAS, CardioMetabochip – GCST90002244 – Eurpoean
Sliz ad – GWAS – GCST90027161 – European
Ferreira atopy – GWAS – GCST005038 – European
Demenais asthma – GWAS – GCST005212 – Multi-ancestry
Liu CD – GWAS – GCST003044 – European
Liu IBD – GWAS – GCST003043 – European
DeLange IBD – ImmunoChip – GCST004131 – European
DeLange CD – ImmunoChip – GCST004132 – European
Tsoi PSO – ImmunoChip – GCST005527 – European
Dubois CeD – GWAS – GCST000612 – European
Hinks JIA – ImmunoChip – GCST005528 – European

'''

'''
First attempt:
In terms of what to do with palindromes,
We will keep hm_code 10-13 and drop 1-9, 14-18.
could consider including 5-9 if lots of drop out.
'''
#files
sumstatspath = '../SumStats/'

'''
Study #0: Barrett JC, type 1 diabetes
All hm_codes are 14, so not harmonized
http://ftp.ebi.ac.uk/pub/databases/gwas/summary_statistics/GCST000001-GCST001000/GCST000392/harmonised/19430480-GCST000392-EFO_0001359.h.tsv.gz


BarrettJC = sumstatspath + "19430480-GCST000392-EFO_0001359.h.tsv"
barrett_t1d = pd.read_csv(BarrettJC, lineterminator='\n', sep='\t')
barrett_t1d.columns.to_list()

barrett_t1d.isna().sum()
pd.unique(barrett_t1d.hm_code)
'''
'''
Study #1: Sawcer, Multiple Sclerosis
http://ftp.ebi.ac.uk/pub/databases/gwas/summary_statistics/GCST001001-GCST002000/GCST001198/harmonised/21833088-GCST001198-EFO_0003885.h.tsv.gz
'''
Sawcer = sumstatspath + "21833088-GCST001198-EFO_0003885.h.tsv"
sawcer_ms = pd.read_csv(Sawcer, lineterminator='\n', sep='\t')
sawcer_ms.columns.to_list()

sawcer_ms.isna().sum()
pd.unique(sawcer_ms.hm_code)

'''
>>> sawcer_ms.isna().sum()
hm_variant_id                    528
hm_rsid                          528
hm_chrom                         528
hm_pos                           528
hm_other_allele                  528
hm_effect_allele                 528
hm_beta                         9037
hm_odds_ratio                   8252
hm_ci_lower                     8282
hm_ci_upper                     8282
hm_effect_allele_frequency    472077
hm_code                            0
chromosome                         0
base_pair_location                 0
variant_id                         0
other_allele                       0
effect_allele                      0
p_value                            0
beta                            7729
standard_error                  7759
odds_ratio                      7729
ci_lower                        7759
ci_upper                        7759
effect_allele_frequency\r          0
dtype: int64
>>> pd.unique(sawcer_ms.hm_code)
array([10, 12, 11, 13, 15])
'''

#Drop hm_variant_id NA
#Drop hm_BETA NA
#DROP hm_odds_ratio NA
sawcer_columns = ['hm_variant_id', 'hm_rsid', 'hm_chrom', 'hm_pos', 'hm_other_allele', 'hm_effect_allele', 'hm_beta', 'hm_odds_ratio', 'hm_ci_lower', 'hm_ci_upper', 'hm_code', 'p_value', 'standard_error']
'''
sawcer_ms
          hm_variant_id     hm_rsid hm_chrom       hm_pos hm_other_allele hm_effect_allele   hm_beta  hm_odds_ratio  ...  effect_allele   p_value      beta  standard_error  odds_ratio  ci_lower  ci_upper effect_allele_frequency\r
0          10_67994_A_C  rs10904494       10      67994.0               A                C  0.024595         1.0249  ...              C  0.188500  0.024595        0.018703      1.0249  0.988010  1.063168                      NA\r
1         10_106057_T_C   rs9286070       10     106057.0               T                C  0.008861         1.0089  ...              G  0.859300  0.008861        0.049985      1.0089  0.914745  1.112747                      NA\r
2         10_113006_C_T  rs11253562       10     113006.0               C                T  0.028976         1.0294  ...              A  0.127700  0.028976        0.019023      1.0294  0.991726  1.068505                      NA\r
3         10_113136_G_A   rs4881551       10     113136.0               G                A -0.011061         0.9890  ...              A  0.545500 -0.011061        0.018297      0.9890  0.954160  1.025112                      NA\r
4         10_113464_G_A   rs4880750       10     113464.0               G                A -0.034074         0.9665  ...              A  0.075810 -0.034074        0.019191      0.9665  0.930821  1.003546                      NA\r
...                 ...         ...      ...          ...             ...              ...       ...            ...  ...            ...       ...       ...             ...         ...       ...       ...                       ...
472072  X_155547903_C_A   rs5940484        X  155547903.0               C                A       NaN            NaN  ...              C  0.013649       NaN             NaN         NaN       NaN       NaN                      NA\r
472073  X_155600716_C_T   rs5983658        X  155600716.0               C                T       NaN            NaN  ...              G  0.010217       NaN             NaN         NaN       NaN       NaN                      NA\r
472074  X_155662569_A_G    rs553678        X  155662569.0               A                G       NaN            NaN  ...              A  0.010655       NaN             NaN         NaN       NaN       NaN                      NA\r
472075  X_155670185_A_G    rs473491        X  155670185.0               A                G       NaN            NaN  ...              A  0.023328       NaN             NaN         NaN       NaN       NaN                      NA\r
472076  X_155699751_C_T    rs557132        X  155699751.0               C                T       NaN            NaN  ...              A  0.019421       NaN             NaN         NaN       NaN       NaN                      NA\r
'''
#Drop hm_variant_id NA
#Drop hm_BETA NA
#DROP hm_odds_ratio NA

sawcer_ms.dropna(axis=0, subset=['hm_variant_id', 'hm_beta', 'hm_odds_ratio'], inplace=True)
'''
sawcer_ms
          hm_variant_id     hm_rsid hm_chrom       hm_pos hm_other_allele hm_effect_allele   hm_beta  hm_odds_ratio  ...  effect_allele  p_value      beta  standard_error  odds_ratio  ci_lower  ci_upper effect_allele_frequency\r
0          10_67994_A_C  rs10904494       10      67994.0               A                C  0.024595       1.024900  ...              C  0.18850  0.024595        0.018703      1.0249  0.988010  1.063168                      NA\r
1         10_106057_T_C   rs9286070       10     106057.0               T                C  0.008861       1.008900  ...              G  0.85930  0.008861        0.049985      1.0089  0.914745  1.112747                      NA\r
2         10_113006_C_T  rs11253562       10     113006.0               C                T  0.028976       1.029400  ...              A  0.12770  0.028976        0.019023      1.0294  0.991726  1.068505                      NA\r
3         10_113136_G_A   rs4881551       10     113136.0               G                A -0.011061       0.989000  ...              A  0.54550 -0.011061        0.018297      0.9890  0.954160  1.025112                      NA\r
4         10_113464_G_A   rs4880750       10     113464.0               G                A -0.034074       0.966500  ...              A  0.07581 -0.034074        0.019191      0.9665  0.930821  1.003546                      NA\r
...                 ...         ...      ...          ...             ...              ...       ...            ...  ...            ...      ...       ...             ...         ...       ...       ...                       ...
464342  9_138116005_C_A   rs2739257        9  138116005.0               C                A -0.016562       0.983574  ...              C  0.56250  0.016562        0.028598      1.0167  0.961279  1.075316                      NA\r
464343  9_138117129_C_T   rs2606357        9  138117129.0               C                T -0.014200       0.985900  ...              A  0.42220 -0.014200        0.017693      0.9859  0.952297  1.020688                      NA\r
464344  9_138117319_A_G   rs2606358        9  138117319.0               A                G -0.020978       0.979240  ...              A  0.23360  0.020978        0.017612      1.0212  0.986550  1.057067                      NA\r
464345  9_138122900_C_T   rs3750510        9  138122900.0               C                T -0.031789       0.968711  ...              G  0.24050  0.031789        0.027084      1.0323  0.978930  1.088579                      NA\r
467618   X_51923690_A_G     rs14115        X   51923690.0               A                G  0.018331       1.018500  ...              G  0.62090  0.018331        0.037064      1.0185  0.947134  1.095243                      NA\r
'''

#Drop invalid hm_code
print(len(sawcer_ms.index))
463040
hm_code_mask = sawcer_ms.hm_code.eq(15)
sawcer_ms.drop(sawcer_ms[hm_code_mask].index, inplace=True)
print(len(sawcer_ms.index))
463040
#did not actually drop anything, since those variants were filtered out with NA dropping.

remap_dict = {'X':23}
sawcer_ms['hm_chrom'] = sawcer_ms['hm_chrom'].replace(remap_dict)
sawcer_ms['hm_chrom'] = sawcer_ms['hm_chrom'].astype('int64')
sawcer_ms['hm_pos'] = sawcer_ms['hm_pos'].astype('int64')
sawcer_ms = sawcer_ms[sawcer_columns].sort_values(by=['hm_chrom', 'hm_pos'])
'''
sawcer_ms
          hm_variant_id     hm_rsid  hm_chrom    hm_pos hm_other_allele hm_effect_allele   hm_beta  hm_odds_ratio  hm_ci_lower  hm_ci_upper  p_value
168307     1_833068_G_A  rs12562034         1    833068               G                A -0.023985       0.976300     0.924095     1.031454   0.3923
168308    1_1070426_C_T   rs3934834         1   1070426               C                T  0.003793       1.003800     0.959020     1.050670   0.8706
168309    1_1083324_A_G   rs9442372         1   1083324               A                G  0.012073       1.012146     0.977618     1.047892   0.4954
168310    1_1086035_A_G   rs3737728         1   1086035               A                G  0.027577       1.027961     0.989888     1.067498   0.1521
168311    1_1095185_C_T   rs6687776         1   1095185               C                T  0.004390       1.004400     0.959126     1.051811   0.8520
...                 ...         ...       ...       ...             ...              ...       ...            ...          ...          ...      ...
230954  22_50724710_C_T    rs715586        22  50724710               C                T -0.003005       0.997000     0.949875     1.046463   0.9032
230955  22_50727236_G_A   rs8137951        22  50727236               G                A  0.015873       1.016000     0.978389     1.055057   0.4095
230956  22_50733265_G_A    rs756638        22  50733265               G                A -0.002002       0.998000     0.959760     1.037763   0.9200
230957  22_50737198_A_G   rs3810648        22  50737198               A                G -0.050136       0.951100     0.886425     1.020494   0.1629
467618   X_51923690_A_G     rs14115        23  51923690               A                G  0.018331       1.018500     0.947134     1.095243   0.6209
'''

sawcer_ms.equals(sawcer_ms.sort_values(by=['hm_chrom', 'hm_pos']))
oc_sawcer = ['hm_chrom', 'hm_pos','hm_other_allele', 'hm_effect_allele', 'hm_beta', 'p_value']

oc_metal = ['hm_variant_id', 'hm_other_allele', 'hm_effect_allele', 'hm_beta', 'standard_error', 'p_value']
sawcer_ms[oc_metal].to_csv('sawcer_ms_gwas_metal-'+version+'.txt', sep='\t', index=False)

'''
#this is for the locus zoom upload
sawcer_ms[oc_sawcer].to_csv('sawcer_ms_gwas_lz'+version+'.txt', sep='\t', index=False)
os.system("gzip -k " + 'sawcer_ms_gwas_lz'+version+'.txt &')
'''
'''
Study #2: Cordell, Primary Biliary Cholangitis
http://ftp.ebi.ac.uk/pub/databases/gwas/summary_statistics/GCST003001-GCST004000/GCST003129/harmonised/26394269-GCST003129-EFO_1001486.h.tsv.gz
'''
Cordell = sumstatspath + "26394269-GCST003129-EFO_1001486.h.tsv"
cordell_pbc = pd.read_csv(Cordell, lineterminator='\n', sep='\t')
cordell_pbc.columns.to_list()

cordell_pbc.isna().sum()
pd.unique(cordell_pbc.hm_code)

'''
>>> cordell_pbc.isna().sum()
hm_variant_id                    8845
hm_rsid                          8845
hm_chrom                         8845
hm_pos                           8845
hm_other_allele                  8845
hm_effect_allele                 8845
hm_beta                          9885
hm_odds_ratio                    8845
hm_ci_lower                      8846
hm_ci_upper                      8846
hm_effect_allele_frequency    1134126
hm_code                             0
chromosome                          0
base_pair_location                  0
variant_id                          0
other_allele                     7216
effect_allele                    7216
p_value                             0
beta                                0
standard_error                      1
odds_ratio                          0
ci_lower                            1
ci_upper                            1
effect_allele_frequency\r           0
dtype: int64
>>> pd.unique(cordell_pbc.hm_code)
array([14, 10, 11,  6,  5, 15, 12, 13])
'''
cordell_columns = ['hm_variant_id', 'hm_rsid', 'hm_chrom', 'hm_pos', 'hm_other_allele', 'hm_effect_allele', 'hm_beta', 'hm_odds_ratio', 'hm_ci_lower', 'hm_ci_upper', 'p_value', 'standard_error']

'''
cordell_pbc
           hm_variant_id     hm_rsid  hm_chrom       hm_pos hm_other_allele hm_effect_allele   hm_beta  hm_odds_ratio  ...  effect_allele   p_value      beta  standard_error  odds_ratio  ci_lower  ci_upper effect_allele_frequency\r
0                    NaN         NaN       NaN          NaN             NaN              NaN       NaN            NaN  ...            NaN  0.933821 -0.004510        0.054314      0.9955  0.894969  1.107323                      NA\r
1           10_52541_A_C  rs12255619      10.0      52541.0               A                C  0.032467       1.033000  ...              C  0.669908  0.032467        0.076165      1.0330  0.889749  1.199315                      NA\r
2           10_58487_T_C  rs11252546      10.0      58487.0               T                C  0.008167       1.008200  ...              C  0.813111  0.008167        0.034543      1.0082  0.942199  1.078824                      NA\r
3           10_66015_A_G   rs7909677      10.0      66015.0               A                G  0.030917       1.031400  ...              G  0.684920  0.030917        0.076196      1.0314  0.888317  1.197530                      NA\r
4           10_67994_A_C  rs10904494      10.0      67994.0               A                C  0.008067       1.008100  ...              C  0.817019  0.008067        0.034866      1.0081  0.941510  1.079400                      NA\r
...                  ...         ...       ...          ...             ...              ...       ...            ...  ...            ...       ...       ...             ...         ...       ...       ...                       ...
1134121  9_138117612_A_G   rs2739260       9.0  138117612.0               A                G  0.008435       1.008471  ...              A  0.804128 -0.008435        0.034013      0.9916  0.927650  1.059959                      NA\r
1134122  9_138122788_C_T   rs3750508       9.0  138122788.0               C                T -0.050241       0.951000  ...              T  0.209698 -0.050241        0.040052      0.9510  0.879200  1.028664                      NA\r
1134123  9_138122891_C_T   rs3750509       9.0  138122891.0               C                T -0.003095       0.996910  ...              C  0.933451  0.003095        0.037067      1.0031  0.932808  1.078688                      NA\r
1134124              NaN         NaN       NaN          NaN             NaN              NaN       NaN            NaN  ...            NaN  0.467970  0.034401        0.047399      1.0350  0.943178  1.135761                      NA\r
1134125              NaN         NaN       NaN          NaN             NaN              NaN       NaN            NaN  ...            NaN  0.829320 -0.015114        0.070109      0.9850  0.858536  1.130093                      NA\r

[1134126 rows x 24 columns]
'''
#Drop hm_variant_id NA
#Drop hm_BETA NA
#DROP hm_odds_ratio NA
cordell_pbc.dropna(axis=0, subset=['hm_variant_id', 'hm_beta', 'hm_odds_ratio'], inplace=True)
'''
cordell_pbc
           hm_variant_id     hm_rsid  hm_chrom       hm_pos hm_other_allele hm_effect_allele   hm_beta  hm_odds_ratio  ...  effect_allele   p_value      beta  standard_error  odds_ratio  ci_lower  ci_upper effect_allele_frequency\r
1           10_52541_A_C  rs12255619      10.0      52541.0               A                C  0.032467       1.033000  ...              C  0.669908  0.032467        0.076165      1.0330  0.889749  1.199315                      NA\r
2           10_58487_T_C  rs11252546      10.0      58487.0               T                C  0.008167       1.008200  ...              C  0.813111  0.008167        0.034543      1.0082  0.942199  1.078824                      NA\r
3           10_66015_A_G   rs7909677      10.0      66015.0               A                G  0.030917       1.031400  ...              G  0.684920  0.030917        0.076196      1.0314  0.888317  1.197530                      NA\r
4           10_67994_A_C  rs10904494      10.0      67994.0               A                C  0.008067       1.008100  ...              C  0.817019  0.008067        0.034866      1.0081  0.941510  1.079400                      NA\r
5           10_80130_C_T  rs11591988      10.0      80130.0               C                T -0.003606       0.996400  ...              T  0.946496 -0.003606        0.053742      0.9964  0.896784  1.107082                      NA\r
...                  ...         ...       ...          ...             ...              ...       ...            ...  ...            ...       ...       ...             ...         ...       ...       ...                       ...
1134119  9_138116189_A_G   rs2739258       9.0  138116189.0               A                G  0.009949       1.009999  ...              A  0.766735 -0.009949        0.033539      0.9901  0.927107  1.057373                      NA\r
1134120  9_138117129_C_T   rs2606357       9.0  138117129.0               C                T -0.002098       0.997904  ...              C  0.952036  0.002098        0.034876      1.0021  0.935888  1.072996                      NA\r
1134121  9_138117612_A_G   rs2739260       9.0  138117612.0               A                G  0.008435       1.008471  ...              A  0.804128 -0.008435        0.034013      0.9916  0.927650  1.059959                      NA\r
1134122  9_138122788_C_T   rs3750508       9.0  138122788.0               C                T -0.050241       0.951000  ...              T  0.209698 -0.050241        0.040052      0.9510  0.879200  1.028664                      NA\r
1134123  9_138122891_C_T   rs3750509       9.0  138122891.0               C                T -0.003095       0.996910  ...              C  0.933451  0.003095        0.037067      1.0031  0.932808  1.078688                      NA\r

[1124241 rows x 24 columns]
'''

#Drop invalid hm_codes
print(len(cordell_pbc.index))
1124241
hm_code_mask = cordell_pbc.hm_code.ge(14) | cordell_pbc.hm_code.le(6)
hm_code_mask.sum()
86500
cordell_pbc.drop(cordell_pbc[hm_code_mask].index, inplace=True)
print(len(cordell_pbc.index))
1037741

remap_dict = {'X':23}
cordell_pbc['hm_chrom'] = cordell_pbc['hm_chrom'].replace(remap_dict)
cordell_pbc['hm_chrom'] = cordell_pbc['hm_chrom'].astype('int64')
cordell_pbc['hm_pos'] = cordell_pbc['hm_pos'].astype('int64')
cordell_pbc = cordell_pbc[cordell_columns].sort_values(by=['hm_chrom', 'hm_pos'])

'''
cordell_pbc
          hm_variant_id    hm_rsid  hm_chrom    hm_pos hm_other_allele hm_effect_allele   hm_beta  hm_odds_ratio  hm_ci_lower  hm_ci_upper  hm_code   p_value
406227    1_1063015_A_G  rs7526076         1   1063015               A                G -0.005087       0.994926     0.915223     1.081570       11  0.904953
406228    1_1070426_C_T  rs3934834         1   1070426               C                T -0.027577       0.972800     0.889891     1.063434       10  0.544008
406229    1_1081817_C_T  rs3766192         1   1081817               C                T  0.013896       1.013993     0.947967     1.084618       11  0.685839
406230    1_1082207_C_T  rs3766191         1   1082207               C                T -0.031491       0.969000     0.882612     1.063843       10  0.508623
406231    1_1083324_A_G  rs9442372         1   1083324               A                G  0.013288       1.013377     0.948607     1.082569       11  0.693345
...                 ...        ...       ...       ...             ...              ...       ...            ...          ...          ...      ...       ...
560295  22_50724710_C_T   rs715586        22  50724710               C                T  0.027712       1.028100     0.940378     1.124005       10  0.542506
560296  22_50727236_G_A  rs8137951        22  50727236               G                A  0.039221       1.040000     0.970716     1.114229       10  0.264836
560297  22_50733265_G_A   rs756638        22  50733265               G                A -0.024600       0.975700     0.908225     1.048187       10  0.501060
560298  22_50737198_A_G  rs3810648        22  50737198               A                G -0.002704       0.997300     0.869699     1.143622       10  0.969124
560299  22_50739662_G_A  rs2285395        22  50739662               G                A -0.005415       0.994600     0.859446     1.151008       10  0.942074

[1037741 rows x 12 columns]
'''

cordell_pbc.equals(cordell_pbc.sort_values(by=['hm_chrom', 'hm_pos']))
oc_cordell = ['hm_chrom', 'hm_pos','hm_other_allele', 'hm_effect_allele', 'hm_beta', 'p_value']

oc_metal = ['hm_variant_id', 'hm_other_allele', 'hm_effect_allele', 'hm_beta', 'standard_error', 'p_value']
cordell_pbc[oc_metal].to_csv('cordell_pbc_gwas_metal-'+version+'.txt', sep='\t', index=False)


'''
#this is for locus zoom upload
cordell_pbc[oc_cordell].to_csv('cordell_pbc_gwas_lz'+version+'.txt', sep='\t', index=False)
os.system("gzip -k " + 'cordell_pbc_gwas_lz'+version+'.txt &')
'''

'''
Study #3: Bentham, Systemic Lupus Erythematosus
http://ftp.ebi.ac.uk/pub/databases/gwas/summary_statistics/GCST003001-GCST004000/GCST003156/harmonised/26502338-GCST003156-EFO_0002690.h.tsv.gz
'''
Bentham = sumstatspath + "26502338-GCST003156-EFO_0002690.h.tsv"
bentham_sle = pd.read_csv(Bentham, lineterminator='\n', sep='\t')
bentham_sle.columns.to_list()

bentham_sle.isna().sum()
pd.unique(bentham_sle.hm_code)

'''
bentham_sle.isna().sum()
hm_variant_id                   19731
hm_rsid                         19731
hm_chrom                        19731
hm_pos                          19731
hm_other_allele                 19731
hm_effect_allele                19731
hm_beta                        843661
hm_odds_ratio                  195773
hm_ci_lower                    195773
hm_ci_upper                    195773
hm_effect_allele_frequency    7914824
hm_code                             0
chromosome                          0
base_pair_location                  0
variant_id                          0
other_allele                        0
effect_allele                       0
p_value                             0
beta                           176354
standard_error                 176354
odds_ratio                     176354
ci_lower                       176354
ci_upper                       176354
effect_allele_frequency\r           0
dtype: int64
>>> pd.unique(bentham_sle.hm_code)
array([15, 10,  5, 11,  6, 12, 13, 14])
'''
bentham_columns = ['hm_variant_id', 'hm_rsid', 'hm_chrom', 'hm_pos', 'hm_other_allele', 'hm_effect_allele', 'hm_beta', 'hm_odds_ratio', 'hm_ci_lower', 'hm_ci_upper', 'p_value', 'standard_error']

'''
bentham_sle
           hm_variant_id      hm_rsid hm_chrom       hm_pos hm_other_allele hm_effect_allele   hm_beta  hm_odds_ratio  ...  effect_allele   p_value      beta  standard_error  odds_ratio  ci_lower  ci_upper effect_allele_frequency\r
0                    NaN          NaN      NaN          NaN             NaN              NaN       NaN            NaN  ...              T  0.398222  0.067659        0.080088        1.07  0.914558  1.251862                      NA\r
1           10_21242_C_T  rs182646175       10      21242.0               C                T -0.010050           0.99  ...              T  0.920081 -0.010050        0.100171        0.99  0.813519  1.204766                      NA\r
2           10_29857_G_T  rs140638708       10      29857.0               G                T  0.039221           1.04  ...              T  0.291772  0.039221        0.037203        1.04  0.966865  1.118667                      NA\r
3           10_31644_A_G   rs28439871       10      31644.0               A                G -0.010050           0.99  ...              G  0.692530 -0.010050        0.025417        0.99  0.941890  1.040567                      NA\r
4                    NaN          NaN      NaN          NaN             NaN              NaN       NaN            NaN  ...              G  0.246440 -0.051293        0.044255        0.95  0.871070  1.036082                      NA\r
...                  ...          ...      ...          ...             ...              ...       ...            ...  ...            ...       ...       ...             ...         ...       ...       ...                       ...
7914819  X_156003433_C_T    rs2037999        X  156003433.0               C                T       NaN           1.00  ...              C  0.140000  0.000000        0.000000        1.00  1.000000  1.000000                      NA\r
7914820  X_156005065_C_T    rs3093505        X  156005065.0               C                T       NaN           1.00  ...              C  0.010000  0.000000        0.000000        1.00  1.000000  1.000000                      NA\r
7914821  X_156006004_C_T    rs3093510        X  156006004.0               C                T       NaN           1.00  ...              T  0.450000  0.000000        0.000000        1.00  1.000000  1.000000                      NA\r
7914822  X_156007058_A_G    rs3093526        X  156007058.0               A                G       NaN           1.00  ...              G  0.070000  0.000000        0.000000        1.00  1.000000  1.000000                      NA\r
7914823  X_156008488_C_T  rs186430584        X  156008488.0               C                T       NaN           1.00  ...              T  0.430000  0.000000        0.000000        1.00  1.000000  1.000000                      NA\r

[7914824 rows x 24 columns]
'''
#Drop hm_variant_id NA
#Drop hm_BETA NA
#DROP hm_odds_ratio NA
bentham_sle.dropna(axis=0, subset=['hm_variant_id', 'hm_beta', 'hm_odds_ratio'], inplace=True)

'''
bentham_sle
           hm_variant_id      hm_rsid hm_chrom       hm_pos hm_other_allele hm_effect_allele   hm_beta  hm_odds_ratio  ...  effect_allele   p_value      beta  standard_error  odds_ratio  ci_lower  ci_upper effect_allele_frequency\r
1           10_21242_C_T  rs182646175       10      21242.0               C                T -0.010050       0.990000  ...              T  0.920081 -0.010050        0.100171        0.99  0.813519  1.204766                      NA\r
2           10_29857_G_T  rs140638708       10      29857.0               G                T  0.039221       1.040000  ...              T  0.291772  0.039221        0.037203        1.04  0.966865  1.118667                      NA\r
3           10_31644_A_G   rs28439871       10      31644.0               A                G -0.010050       0.990000  ...              G  0.692530 -0.010050        0.025417        0.99  0.941890  1.040567                      NA\r
6           10_47876_C_T    rs9329305       10      47876.0               C                T  0.039221       1.040000  ...              T  0.234780  0.039221        0.033010        1.04  0.974842  1.109513                      NA\r
8           10_48196_C_T    rs4607995       10      48196.0               C                T  0.030459       1.030928  ...              C  0.456923 -0.030459        0.040944        0.97  0.895199  1.051051                      NA\r
...                  ...          ...      ...          ...             ...              ...       ...            ...  ...            ...       ...       ...             ...         ...       ...       ...                       ...
7733648  9_138174168_A_T   rs10780202        9  138174168.0               A                T -0.019803       0.980392  ...              A  0.602465  0.019803        0.038019        1.02  0.946755  1.098912                      NA\r
7733650  9_138174185_G_A    rs4088486        9  138174185.0               G                A  0.048790       1.050000  ...              A  0.481173  0.048790        0.069263        1.05  0.916709  1.202672                      NA\r
7733651  9_138174670_T_G   rs71512810        9  138174670.0               T                G -0.105361       0.900000  ...              G  0.375200 -0.105361        0.118813        0.90  0.713028  1.136000                      NA\r
7733652  X_101529224_C_A    rs5951348        X  101529224.0               C                A -0.009950       0.990099  ...              C  0.881848  0.009950        0.066948        1.01  0.885796  1.151619                      NA\r
7733653   X_46396671_G_C   rs12007097        X   46396671.0               G                C -0.048790       0.952381  ...              G  0.567067  0.048790        0.085241        1.05  0.888445  1.240933                      NA\r

[7071163 rows x 24 columns]
'''

#Drop invalid hm_codes
print(len(bentham_sle.index))
7071163
hm_code_mask = bentham_sle.hm_code.ge(14) | bentham_sle.hm_code.le(6)
hm_code_mask.sum()
1024080
bentham_sle.drop(bentham_sle[hm_code_mask].index, inplace=True)
print(len(bentham_sle.index))
6047083

'''
bentham_sle
           hm_variant_id      hm_rsid hm_chrom       hm_pos hm_other_allele hm_effect_allele   hm_beta  hm_odds_ratio  ...  effect_allele   p_value      beta  standard_error  odds_ratio  ci_lower  ci_upper effect_allele_frequency\r
1           10_21242_C_T  rs182646175       10      21242.0               C                T -0.010050       0.990000  ...              T  0.920081 -0.010050        0.100171        0.99  0.813519  1.204766                      NA\r
2           10_29857_G_T  rs140638708       10      29857.0               G                T  0.039221       1.040000  ...              T  0.291772  0.039221        0.037203        1.04  0.966865  1.118667                      NA\r
3           10_31644_A_G   rs28439871       10      31644.0               A                G -0.010050       0.990000  ...              G  0.692530 -0.010050        0.025417        0.99  0.941890  1.040567                      NA\r
6           10_47876_C_T    rs9329305       10      47876.0               C                T  0.039221       1.040000  ...              T  0.234780  0.039221        0.033010        1.04  0.974842  1.109513                      NA\r
8           10_48196_C_T    rs4607995       10      48196.0               C                T  0.030459       1.030928  ...              C  0.456923 -0.030459        0.040944        0.97  0.895199  1.051051                      NA\r
...                  ...          ...      ...          ...             ...              ...       ...            ...  ...            ...       ...       ...             ...         ...       ...       ...                       ...
7733646  9_138172038_C_T   rs62581014        9  138172038.0               C                T  0.009950       1.010000  ...              T  0.799274  0.009950        0.039130        1.01  0.935434  1.090510                      NA\r
7733647  9_138172039_G_A    rs9314655        9  138172039.0               G                A  0.039221       1.040000  ...              A  0.438054  0.039221        0.050576        1.04  0.941851  1.148377                      NA\r
7733650  9_138174185_G_A    rs4088486        9  138174185.0               G                A  0.048790       1.050000  ...              A  0.481173  0.048790        0.069263        1.05  0.916709  1.202672                      NA\r
7733651  9_138174670_T_G   rs71512810        9  138174670.0               T                G -0.105361       0.900000  ...              G  0.375200 -0.105361        0.118813        0.90  0.713028  1.136000                      NA\r
7733652  X_101529224_C_A    rs5951348        X  101529224.0               C                A -0.009950       0.990099  ...              C  0.881848  0.009950        0.066948        1.01  0.885796  1.151619                      NA\r

[6047083 rows x 24 columns]
'''

#need to type cast, etc.

remap_dict = {'X':23}
bentham_sle['hm_chrom'] = bentham_sle['hm_chrom'].replace(remap_dict)
bentham_sle['hm_chrom'] = bentham_sle['hm_chrom'].astype('int64')
bentham_sle['hm_pos'] = bentham_sle['hm_pos'].astype('int64')
bentham_sle = bentham_sle[bentham_columns].sort_values(by=['hm_chrom', 'hm_pos'])

'''
bentham_sle
           hm_variant_id       hm_rsid  hm_chrom     hm_pos hm_other_allele hm_effect_allele   hm_beta  hm_odds_ratio  hm_ci_lower  hm_ci_upper   p_value
2769473     1_832873_A_C     rs2977608         1     832873               A                C -0.019803       0.980392     0.905889     1.061022  0.623366
2769474     1_846465_A_G    rs61768199         1     846465               A                G  0.009950       1.010000     0.877822     1.162081  0.889415
2769475     1_852226_G_T     rs3863622         1     852226               G                T -0.010050       0.990000     0.882073     1.111133  0.864496
2769477     1_855085_G_A    rs61768207         1     855085               G                A  0.019803       1.020000     0.901576     1.153979  0.753144
2769478     1_860995_T_C    rs12083781         1     860995               T                C  0.048790       1.050000     0.948177     1.162757  0.348501
...                  ...           ...       ...        ...             ...              ...       ...            ...          ...          ...       ...
3758614  22_50783303_T_C   rs115055839        22   50783303               T                C  0.029559       1.030000     0.922267     1.150317  0.600000
3758615  22_50783672_G_T  rs1321244694        22   50783672               G                T -0.010050       0.990000     0.791129     1.238862  0.930000
3758616  22_50784338_G_A  rs1362912151        22   50784338               G                A  0.009950       1.010000     0.916999     1.112432  0.840000
3758618  22_50791377_T_C     rs9616985        22   50791377               T                C  0.029559       1.030000     0.913285     1.161630  0.630000
7733652  X_101529224_C_A     rs5951348        23  101529224               C                A -0.009950       0.990099     0.868342     1.128928  0.881848

[6047083 rows x 11 columns]
'''

'''
filter balloons
'''

#filter out the bentham balloons
balloons1 = ["rs7285053", "rs8078864", "rs17091347", "rs73050535", "rs1034009", "rs10264693", "rs17168663", "rs16895550", "rs9969061", "rs11962557", "rs13170409", "rs6532924", "rs17087866", "rs12081621", "rs4074976", "rs4661543", "rs13019891", "rs512681", "rs10200680", "rs2573219","rs9852014", "rs1464446", "rs1078324", "rs7386188", "rs7823055", "rs11928304", "rs12309414", "rs12948819"]
balloons2 = ["rs34703115","rs55684314","rs77601723","rs73068668","20:7357645:i","rs111478576","rs1120809","rs11905662","rs11907821","rs2423194","rs34995148","rs56412650","rs57830939","rs58072943","rs58605847","rs58829623","rs60461307","rs7262084","rs73894580","rs73894583","rs73894585","rs73894599","rs73897155","rs77222522"]
balloons = balloons1+balloons2
print("# of balloons: " + str(len(balloons)))
balloonmask = bentham_sle.hm_rsid.isin(balloons)
bentham_sle = bentham_sle[~balloonmask]
print("rows after balloon filtering: "+ str(len(bentham_sle.index)))
#11963223


bentham_sle.equals(bentham_sle.sort_values(by=['hm_chrom', 'hm_pos']))
oc_bentham = ['hm_chrom', 'hm_pos','hm_other_allele', 'hm_effect_allele', 'hm_beta', 'p_value']

oc_metal = ['hm_variant_id', 'hm_other_allele', 'hm_effect_allele', 'hm_beta', 'standard_error', 'p_value']
bentham_sle[oc_metal].to_csv('bentham_sle_gwas_metal-'+version+'.txt', sep='\t', index=False)

'''
#for locus zoom upload
bentham_sle[oc_bentham].to_csv('bentham_sle_gwas_lz'+version+'.txt', sep='\t', index=False)
os.system("gzip -k " + 'bentham_sle_gwas_lz'+version+'.txt &')
'''

'''
Study #4: Bronson, et al. GWAS of selective IgA deficiency sIgA
– could not resolve whether harmonised correctly, so did not include.
https://www.ebi.ac.uk/gwas/studies/GCST003814
https://www.ebi.ac.uk/gwas/publications/27723758
'''

'''
Study #5: Ji,Primary Sclerosing Cholangitis
http://ftp.ebi.ac.uk/pub/databases/gwas/summary_statistics/GCST004001-GCST005000/GCST004030/harmonised/27992413-GCST004030-EFO_0004268.h.tsv.gz
27992413
'''
Ji = sumstatspath + "27992413-GCST004030-EFO_0004268.h.tsv"
ji_psc = pd.read_csv(Ji, lineterminator='\n', sep='\t')
ji_psc.columns.to_list()

ji_psc.isna().sum()
pd.unique(ji_psc.hm_code)

'''
ji_psc.isna().sum()
hm_variant_id                   74372
hm_rsid                         74372
hm_chrom                        74372
hm_pos                          74372
hm_other_allele                 74372
hm_effect_allele                74372
hm_beta                       7866424
hm_odds_ratio                   74372
hm_ci_lower                   7866424
hm_ci_upper                   7866424
hm_effect_allele_frequency      74399
hm_code                             0
chromosome                          0
variant_id                          0
base_pair_location                  0
other_allele                        0
effect_allele                       0
freq_1                              0
freq_1_cases                        0
effect_allele_frequency             0
mmm_var_info_nonmissing             0
odds_ratio                          0
standard_error                      0
p_value                             0
platform                            0
beta                          7866424
ci_upper                      7866424
ci_lower\r                          0
dtype: int64
>>> pd.unique(ji_psc.hm_code)
array([15,  6, 11,  5, 10, 12, 13, 14])
'''

ji_columns = ['hm_variant_id', 'hm_rsid', 'hm_chrom', 'hm_pos', 'hm_other_allele', 'hm_effect_allele', 'hm_beta', 'hm_odds_ratio', 'hm_ci_lower', 'hm_ci_upper', 'p_value', 'standard_error']

'''
ji_psc
           hm_variant_id      hm_rsid hm_chrom       hm_pos hm_other_allele hm_effect_allele  hm_beta  hm_odds_ratio  ...  mmm_var_info_nonmissing  odds_ratio  standard_error   p_value  platform beta  ci_upper ci_lower\r
0                    NaN          NaN      NaN          NaN             NaN              NaN      NaN            NaN  ...                    0.992       1.050           0.087  0.555758      both  NaN       NaN       NA\r
1           10_46126_A_T  rs138929159       10      46126.0               A                T      NaN       0.976562  ...                    0.984       1.024           0.081  0.763204      both  NaN       NaN       NA\r
2           10_47588_G_A  rs200635479       10      47588.0               G                A      NaN       0.976562  ...                    0.984       1.024           0.081  0.763212      both  NaN       NaN       NA\r
3           10_48219_C_T  rs145359804       10      48219.0               C                T      NaN       0.973710  ...                    0.988       1.027           0.081  0.735534      both  NaN       NaN       NA\r
4           10_48232_G_A   rs12218882       10      48232.0               G                A      NaN       0.907441  ...                    0.947       1.102           0.094  0.281003      both  NaN       NaN       NA\r
...                  ...          ...      ...          ...             ...              ...      ...            ...  ...                      ...         ...             ...       ...       ...  ...       ...        ...
7866419  X_155688581_C_T   rs73562846        X  155688581.0               C                T      NaN       0.988142  ...                    0.982       1.012           0.039  0.764354      omni  NaN       NaN       NA\r
7866420  X_155688722_G_T     rs641588        X  155688722.0               G                T      NaN       0.966184  ...                    1.000       1.035           0.028  0.209204      both  NaN       NaN       NA\r
7866421  X_155695384_C_T     rs509981        X  155695384.0               C                T      NaN       0.964320  ...                    0.999       1.037           0.028  0.184855      both  NaN       NaN       NA\r
7866422  X_155699751_C_T     rs557132        X  155699751.0               C                T      NaN       0.965251  ...                    0.999       1.036           0.028  0.202540      both  NaN       NaN       NA\r
7866423  X_155700569_A_G     rs781880        X  155700569.0               A                G      NaN       0.964320  ...                    0.972       1.037           0.028  0.199268      both  NaN       NaN       NA\r

[7866424 rows x 28 columns]
'''

#Drop hm_variant_id NA
#DROP hm_odds_ratio NA
#NOT DROPPING hm_beta due to ... all of them are NaN
ji_psc.dropna(axis=0, subset=['hm_variant_id', 'hm_odds_ratio'], inplace=True)
ji_psc = ji_psc.assign(hm_beta=lambda x: np.log(x['hm_odds_ratio']))

'''
ji_psc
           hm_variant_id      hm_rsid hm_chrom       hm_pos hm_other_allele hm_effect_allele   hm_beta  hm_odds_ratio  ...  mmm_var_info_nonmissing  odds_ratio  standard_error   p_value  platform beta  ci_upper ci_lower\r
1           10_46126_A_T  rs138929159       10      46126.0               A                T -0.023717       0.976562  ...                    0.984       1.024           0.081  0.763204      both  NaN       NaN       NA\r
2           10_47588_G_A  rs200635479       10      47588.0               G                A -0.023717       0.976562  ...                    0.984       1.024           0.081  0.763212      both  NaN       NaN       NA\r
3           10_48219_C_T  rs145359804       10      48219.0               C                T -0.026642       0.973710  ...                    0.988       1.027           0.081  0.735534      both  NaN       NaN       NA\r
4           10_48232_G_A   rs12218882       10      48232.0               G                A -0.097127       0.907441  ...                    0.947       1.102           0.094  0.281003      both  NaN       NaN       NA\r
5           10_48323_C_A  rs184120752       10      48323.0               C                A -0.026642       0.973710  ...                    0.988       1.027           0.081  0.735525      both  NaN       NaN       NA\r
...                  ...          ...      ...          ...             ...              ...       ...            ...  ...                      ...         ...             ...       ...       ...  ...       ...        ...
7866419  X_155688581_C_T   rs73562846        X  155688581.0               C                T -0.011929       0.988142  ...                    0.982       1.012           0.039  0.764354      omni  NaN       NaN       NA\r
7866420  X_155688722_G_T     rs641588        X  155688722.0               G                T -0.034401       0.966184  ...                    1.000       1.035           0.028  0.209204      both  NaN       NaN       NA\r
7866421  X_155695384_C_T     rs509981        X  155695384.0               C                T -0.036332       0.964320  ...                    0.999       1.037           0.028  0.184855      both  NaN       NaN       NA\r
7866422  X_155699751_C_T     rs557132        X  155699751.0               C                T -0.035367       0.965251  ...                    0.999       1.036           0.028  0.202540      both  NaN       NaN       NA\r
7866423  X_155700569_A_G     rs781880        X  155700569.0               A                G -0.036332       0.964320  ...                    0.972       1.037           0.028  0.199268      both  NaN       NaN       NA\r

[7792052 rows x 28 columns]
'''

#Drop invalid hm_codes
print(len(ji_psc.index))
7792052
hm_code_mask = ji_psc.hm_code.ge(14) | ji_psc.hm_code.le(6)
hm_code_mask.sum()
1084287
ji_psc.drop(ji_psc[hm_code_mask].index, inplace=True)
print(len(ji_psc.index))
6707765

'''
ji_psc
           hm_variant_id      hm_rsid hm_chrom       hm_pos hm_other_allele hm_effect_allele   hm_beta  hm_odds_ratio  ...  mmm_var_info_nonmissing  odds_ratio  standard_error   p_value  platform beta  ci_upper ci_lower\r
2           10_47588_G_A  rs200635479       10      47588.0               G                A -0.023717       0.976562  ...                    0.984       1.024           0.081  0.763212      both  NaN       NaN       NA\r
3           10_48219_C_T  rs145359804       10      48219.0               C                T -0.026642       0.973710  ...                    0.988       1.027           0.081  0.735534      both  NaN       NaN       NA\r
4           10_48232_G_A   rs12218882       10      48232.0               G                A -0.097127       0.907441  ...                    0.947       1.102           0.094  0.281003      both  NaN       NaN       NA\r
5           10_48323_C_A  rs184120752       10      48323.0               C                A -0.026642       0.973710  ...                    0.988       1.027           0.081  0.735525      both  NaN       NaN       NA\r
6           10_48601_C_A   rs11251906       10      48601.0               C                A -0.116894       0.889680  ...                    0.930       1.124           0.097  0.204167      both  NaN       NaN       NA\r
...                  ...          ...      ...          ...             ...              ...       ...            ...  ...                      ...         ...             ...       ...       ...  ...       ...        ...
7866419  X_155688581_C_T   rs73562846        X  155688581.0               C                T -0.011929       0.988142  ...                    0.982       1.012           0.039  0.764354      omni  NaN       NaN       NA\r
7866420  X_155688722_G_T     rs641588        X  155688722.0               G                T -0.034401       0.966184  ...                    1.000       1.035           0.028  0.209204      both  NaN       NaN       NA\r
7866421  X_155695384_C_T     rs509981        X  155695384.0               C                T -0.036332       0.964320  ...                    0.999       1.037           0.028  0.184855      both  NaN       NaN       NA\r
7866422  X_155699751_C_T     rs557132        X  155699751.0               C                T -0.035367       0.965251  ...                    0.999       1.036           0.028  0.202540      both  NaN       NaN       NA\r
7866423  X_155700569_A_G     rs781880        X  155700569.0               A                G -0.036332       0.964320  ...                    0.972       1.037           0.028  0.199268      both  NaN       NaN       NA\r

[6707765 rows x 28 columns]
'''

#need to type cast, etc.
remap_dict = {'X':23}
ji_psc['hm_chrom'] = ji_psc['hm_chrom'].replace(remap_dict)
ji_psc['hm_chrom'] = ji_psc['hm_chrom'].astype('int64')
ji_psc['hm_pos'] = ji_psc['hm_pos'].astype('int64')
ji_psc = ji_psc[ji_columns].sort_values(by=['hm_chrom', 'hm_pos'])

'''
 ji_psc
           hm_variant_id      hm_rsid  hm_chrom     hm_pos hm_other_allele hm_effect_allele   hm_beta  hm_odds_ratio  hm_ci_lower  hm_ci_upper   p_value
2676431     1_812952_T_C  rs182373484         1     812952               T                C -0.073250       0.929368          NaN          NaN  0.696699
2676432     1_826372_C_T    rs1057213         1     826372               C                T -0.022739       0.977517          NaN          NaN  0.582989
2676433     1_828811_T_G    rs7515915         1     828811               T                G  0.024693       1.025000          NaN          NaN  0.556183
2676436     1_836443_T_C    rs2977605         1     836443               T                C -0.023717       0.976562          NaN          NaN  0.558617
2676437     1_836587_G_A   rs59066358         1     836587               G                A  0.027615       1.028000          NaN          NaN  0.504336
...                  ...          ...       ...        ...             ...              ...       ...            ...          ...          ...       ...
7866419  X_155688581_C_T   rs73562846        23  155688581               C                T -0.011929       0.988142          NaN          NaN  0.764354
7866420  X_155688722_G_T     rs641588        23  155688722               G                T -0.034401       0.966184          NaN          NaN  0.209204
7866421  X_155695384_C_T     rs509981        23  155695384               C                T -0.036332       0.964320          NaN          NaN  0.184855
7866422  X_155699751_C_T     rs557132        23  155699751               C                T -0.035367       0.965251          NaN          NaN  0.202540
7866423  X_155700569_A_G     rs781880        23  155700569               A                G -0.036332       0.964320          NaN          NaN  0.199268

[6707765 rows x 11 columns]'''

ji_psc.equals(ji_psc.sort_values(by=['hm_chrom', 'hm_pos']))
oc_ji = ['hm_chrom', 'hm_pos','hm_other_allele', 'hm_effect_allele', 'hm_beta', 'p_value']

oc_metal = ['hm_variant_id', 'hm_other_allele', 'hm_effect_allele', 'hm_beta', 'standard_error', 'p_value']
ji_psc[oc_metal].to_csv('ji_psc_gwas_metal-'+version+'.txt', sep='\t', index=False)


'''
#locus zoom setup
ji_psc[oc_ji].to_csv('ji_psc_gwas_lz'+version+'.txt', sep='\t', index=False)
os.system("gzip -k " + 'ji_psc_gwas_lz'+version+'.txt &')
'''

'''
Study #6: Ferreira, Atopy
http://ftp.ebi.ac.uk/pub/databases/gwas/summary_statistics/GCST005001-GCST006000/GCST005038/harmonised/29083406-GCST005038-EFO_0003785.h.tsv.gz
29083406
'''
Ferreira = sumstatspath + "29083406-GCST005038-EFO_0003785.h.tsv"
ferriera_atopy = pd.read_csv(Ferreira, lineterminator='\n', sep='\t')
ferriera_atopy.columns.to_list()

ferriera_atopy.isna().sum()
pd.unique(ferriera_atopy.hm_code)

'''
ferriera_atopy.isna().sum()
hm_variant_id                   15129
hm_rsid                         15129
hm_chrom                        15129
hm_pos                          15129
hm_other_allele                 15129
hm_effect_allele                15129
hm_beta                         48331
hm_odds_ratio                 8303354
hm_ci_lower                   8303354
hm_ci_upper                   8303354
hm_effect_allele_frequency      15129
hm_code                             0
chromosome                          0
base_pair_location                  0
effect_allele                       0
other_allele                        0
beta                                0
standard_error                      0
p_value                             0
variant_id                          0
effect_allele_frequency             0
odds_ratio                    8303354
ci_lower                      8303354
ci_upper\r                          0
dtype: int64
>>> pd.unique(ferriera_atopy.hm_code)
array([10, 11,  5,  6, 15, 12, 13])
'''

ferriera_columns = ['hm_variant_id', 'hm_rsid', 'hm_chrom', 'hm_pos', 'hm_other_allele', 'hm_effect_allele', 'hm_beta', 'hm_odds_ratio', 'hm_ci_lower', 'hm_ci_upper', 'p_value', 'standard_error']

'''
ferriera_atopy
            hm_variant_id      hm_rsid hm_chrom       hm_pos hm_other_allele hm_effect_allele  hm_beta  hm_odds_ratio  ...    beta  standard_error  p_value   variant_id  effect_allele_frequency  odds_ratio ci_lower ci_upper\r
0          1_99534456_G_T   rs10875231        1   99534456.0               G                T  -0.0072            NaN  ... -0.0072          0.0067   0.2826   rs10875231                  0.24490         NaN      NaN       NA\r
1          1_99535271_C_T    rs6678176        1   99535271.0               C                T  -0.0070            NaN  ... -0.0070          0.0063   0.2656    rs6678176                  0.31970         NaN      NaN       NA\r
2          1_99535287_T_C   rs78286437        1   99535287.0               T                C  -0.0002            NaN  ...  0.0002          0.0131   0.9865   rs78286437                  0.07483         NaN      NaN       NA\r
3        1_99535433_A_ATC  rs146963890        1   99535433.0               A              ATC  -0.0078            NaN  ...  0.0078          0.0138   0.5728  rs146963890                  0.06803         NaN      NaN       NA\r
4          1_99535582_A_G  rs144406489        1   99535582.0               A                G   0.0049            NaN  ... -0.0049          0.0214   0.8196  rs144406489                  0.02041         NaN      NaN       NA\r
...                   ...          ...      ...          ...             ...              ...      ...            ...  ...     ...             ...      ...          ...                      ...         ...      ...        ...
8303349   X_100740828_T_C    rs5921637        X  100740828.0               T                C   0.0024            NaN  ... -0.0024          0.0060   0.6832    rs5921637                  0.45830         NaN      NaN       NA\r
8303350   X_100740944_C_G    rs4828052        X  100740944.0               C                G   0.0023            NaN  ... -0.0023          0.0060   0.7045    rs4828052                  0.46150         NaN      NaN       NA\r
8303351   X_100742417_T_G    rs4828054        X  100742417.0               T                G   0.0025            NaN  ... -0.0025          0.0059   0.6741    rs4828054                  0.46150         NaN      NaN       NA\r
8303352   X_100743450_A_C   rs73559504        X  100743450.0               A                C   0.0287            NaN  ... -0.0287          0.0189   0.1285   rs73559504                  0.02885         NaN      NaN       NA\r
8303353   X_100744360_G_A    rs2021704        X  100744360.0               G                A   0.0007            NaN  ...  0.0007          0.0060   0.9004    rs2021704                  0.46150         NaN      NaN       NA\r

[8303354 rows x 24 columns]
'''

#Drop hm_variant_id NA
#no hm_odds_ratio that is not NaN
#DROP hm_beta NA
#NOT DROPPING hm_beta due to ... all of them are NaN
ferriera_atopy.dropna(axis=0, subset=['hm_variant_id', 'hm_beta'], inplace=True)

'''
ferriera_atopy
            hm_variant_id      hm_rsid hm_chrom       hm_pos hm_other_allele hm_effect_allele  hm_beta  hm_odds_ratio  ...    beta  standard_error  p_value   variant_id  effect_allele_frequency  odds_ratio ci_lower ci_upper\r
0          1_99534456_G_T   rs10875231        1   99534456.0               G                T  -0.0072            NaN  ... -0.0072          0.0067   0.2826   rs10875231                  0.24490         NaN      NaN       NA\r
1          1_99535271_C_T    rs6678176        1   99535271.0               C                T  -0.0070            NaN  ... -0.0070          0.0063   0.2656    rs6678176                  0.31970         NaN      NaN       NA\r
2          1_99535287_T_C   rs78286437        1   99535287.0               T                C  -0.0002            NaN  ...  0.0002          0.0131   0.9865   rs78286437                  0.07483         NaN      NaN       NA\r
3        1_99535433_A_ATC  rs146963890        1   99535433.0               A              ATC  -0.0078            NaN  ...  0.0078          0.0138   0.5728  rs146963890                  0.06803         NaN      NaN       NA\r
4          1_99535582_A_G  rs144406489        1   99535582.0               A                G   0.0049            NaN  ... -0.0049          0.0214   0.8196  rs144406489                  0.02041         NaN      NaN       NA\r
...                   ...          ...      ...          ...             ...              ...      ...            ...  ...     ...             ...      ...          ...                      ...         ...      ...        ...
8303349   X_100740828_T_C    rs5921637        X  100740828.0               T                C   0.0024            NaN  ... -0.0024          0.0060   0.6832    rs5921637                  0.45830         NaN      NaN       NA\r
8303350   X_100740944_C_G    rs4828052        X  100740944.0               C                G   0.0023            NaN  ... -0.0023          0.0060   0.7045    rs4828052                  0.46150         NaN      NaN       NA\r
8303351   X_100742417_T_G    rs4828054        X  100742417.0               T                G   0.0025            NaN  ... -0.0025          0.0059   0.6741    rs4828054                  0.46150         NaN      NaN       NA\r
8303352   X_100743450_A_C   rs73559504        X  100743450.0               A                C   0.0287            NaN  ... -0.0287          0.0189   0.1285   rs73559504                  0.02885         NaN      NaN       NA\r
8303353   X_100744360_G_A    rs2021704        X  100744360.0               G                A   0.0007            NaN  ...  0.0007          0.0060   0.9004    rs2021704                  0.46150         NaN      NaN       NA\r

[8255023 rows x 24 columns]
'''

#Drop invalid hm_codes
print(len(ferriera_atopy.index))
8255023
hm_code_mask = ferriera_atopy.hm_code.ge(14) | ferriera_atopy.hm_code.le(6)
hm_code_mask.sum()
1187356
ferriera_atopy.drop(ferriera_atopy[hm_code_mask].index, inplace=True)
print(len(ferriera_atopy.index))
7067667

'''
ferriera_atopy
            hm_variant_id      hm_rsid hm_chrom       hm_pos hm_other_allele hm_effect_allele  hm_beta  hm_odds_ratio  ...    beta  standard_error  p_value   variant_id  effect_allele_frequency  odds_ratio ci_lower ci_upper\r
0          1_99534456_G_T   rs10875231        1   99534456.0               G                T  -0.0072            NaN  ... -0.0072          0.0067   0.2826   rs10875231                  0.24490         NaN      NaN       NA\r
1          1_99535271_C_T    rs6678176        1   99535271.0               C                T  -0.0070            NaN  ... -0.0070          0.0063   0.2656    rs6678176                  0.31970         NaN      NaN       NA\r
2          1_99535287_T_C   rs78286437        1   99535287.0               T                C  -0.0002            NaN  ...  0.0002          0.0131   0.9865   rs78286437                  0.07483         NaN      NaN       NA\r
3        1_99535433_A_ATC  rs146963890        1   99535433.0               A              ATC  -0.0078            NaN  ...  0.0078          0.0138   0.5728  rs146963890                  0.06803         NaN      NaN       NA\r
4          1_99535582_A_G  rs144406489        1   99535582.0               A                G   0.0049            NaN  ... -0.0049          0.0214   0.8196  rs144406489                  0.02041         NaN      NaN       NA\r
...                   ...          ...      ...          ...             ...              ...      ...            ...  ...     ...             ...      ...          ...                      ...         ...      ...        ...
8303348   X_100738140_G_A  rs142580994        X  100738140.0               G                A  -0.0088            NaN  ... -0.0088          0.0219   0.6888  rs142580994                  0.03205         NaN      NaN       NA\r
8303349   X_100740828_T_C    rs5921637        X  100740828.0               T                C   0.0024            NaN  ... -0.0024          0.0060   0.6832    rs5921637                  0.45830         NaN      NaN       NA\r
8303351   X_100742417_T_G    rs4828054        X  100742417.0               T                G   0.0025            NaN  ... -0.0025          0.0059   0.6741    rs4828054                  0.46150         NaN      NaN       NA\r
8303352   X_100743450_A_C   rs73559504        X  100743450.0               A                C   0.0287            NaN  ... -0.0287          0.0189   0.1285   rs73559504                  0.02885         NaN      NaN       NA\r
8303353   X_100744360_G_A    rs2021704        X  100744360.0               G                A   0.0007            NaN  ...  0.0007          0.0060   0.9004    rs2021704                  0.46150         NaN      NaN       NA\r

[7067667 rows x 24 columns]

'''

#need to type cast, etc.
remap_dict = {'X':23}
ferriera_atopy['hm_chrom'] = ferriera_atopy['hm_chrom'].replace(remap_dict)
ferriera_atopy['hm_chrom'] = ferriera_atopy['hm_chrom'].astype('int64')
ferriera_atopy['hm_pos'] = ferriera_atopy['hm_pos'].astype('int64')
ferriera_atopy = ferriera_atopy[ferriera_columns].sort_values(by=['hm_chrom', 'hm_pos'])

'''
ferriera_atopy
           hm_variant_id      hm_rsid  hm_chrom     hm_pos hm_other_allele hm_effect_allele  hm_beta  hm_odds_ratio  hm_ci_lower  hm_ci_upper  p_value
630998      1_816376_T_C   rs28527770         1     816376               T                C  -0.0053            NaN          NaN          NaN  0.56890
549691     1_816927_AT_A  rs201062411         1     816927              AT                A  -0.0450            NaN          NaN          NaN  0.01366
549775      1_817098_G_A  rs146277091         1     817098               G                A  -0.0431            NaN          NaN          NaN  0.01687
549837      1_817186_G_A    rs3094315         1     817186               G                A   0.0147            NaN          NaN          NaN  0.09130
549854      1_817237_C_A  rs149886465         1     817237               C                A  -0.0425            NaN          NaN          NaN  0.01948
...                  ...          ...       ...        ...             ...              ...      ...            ...          ...          ...      ...
8232880  X_155688581_C_T   rs73562846        23  155688581               C                T  -0.0095            NaN          NaN          NaN  0.32990
8232881  X_155688722_G_T     rs641588        23  155688722               G                T  -0.0083            NaN          NaN          NaN  0.30360
8232883  X_155696234_C_T     rs538470        23  155696234               C                T  -0.0086            NaN          NaN          NaN  0.28910
8232884  X_155697538_C_T     rs645904        23  155697538               C                T  -0.0076            NaN          NaN          NaN  0.34580
8232885  X_155699751_C_T     rs557132        23  155699751               C                T  -0.0083            NaN          NaN          NaN  0.30090

[7067667 rows x 11 columns]
'''
ferriera_atopy.equals(ferriera_atopy.sort_values(by=['hm_chrom', 'hm_pos']))
oc_ferriera = ['hm_chrom', 'hm_pos','hm_other_allele', 'hm_effect_allele', 'hm_beta', 'p_value']

'''
ferriera_atopy[oc_ferriera].to_csv('ferriera_atopy_gwas_lz'+version+'.txt', sep='\t', index=False)
os.system("gzip -k " + 'ferriera_atopy_gwas_lz'+version+'.txt &')
'''

'''
Study #7: Demenais, Asthma
http://ftp.ebi.ac.uk/pub/databases/gwas/summary_statistics/GCST005001-GCST006000/GCST005212/harmonised/29273806-GCST005212-EFO_0000270.h.tsv.gz
29273806
'''
Demenais = sumstatspath + "29273806-GCST005212-EFO_0000270.h.tsv"
demenais_asthma = pd.read_csv(Demenais, lineterminator='\n', sep='\t')
demenais_asthma.columns.to_list()

demenais_asthma.isna().sum()
pd.unique(demenais_asthma.hm_code)

'''
 demenais_asthma.isna().sum()
hm_variant_id                     1994
hm_rsid                           1994
hm_chrom                          1994
hm_pos                            1994
hm_other_allele                   1994
hm_effect_allele                  1994
hm_beta                           1994
hm_odds_ratio                  2001256
hm_ci_lower                    2001256
hm_ci_upper                    2001256
hm_effect_allele_frequency     2001256
hm_code                              0
chromosome                           0
variant_id                           0
base_pair_location                   0
other_allele                         0
effect_allele                        0
beta                                 0
standard_error                       0
p_value                              0
multiancestry_beta_rand              0
multiancestry_se_rand                0
multiancestry_pval_rand              0
multiancestry_hetqtest               0
multiancestry_df_hetqtest            0
multiancestry_pval_hetqtest          0
ci_upper                       2001256
effect_allele_frequency        2001256
odds_ratio                     2001256
ci_lower\r                           0
dtype: int64
>>> pd.unique(demenais_asthma.hm_code)
array([10, 11, 15, 12, 13])
'''

demenais_columns = ['hm_variant_id', 'hm_rsid', 'hm_chrom', 'hm_pos', 'hm_other_allele', 'hm_effect_allele', 'hm_beta', 'hm_odds_ratio', 'hm_ci_lower', 'hm_ci_upper', 'p_value', 'standard_error']

'''
 demenais_asthma
           hm_variant_id     hm_rsid  hm_chrom       hm_pos hm_other_allele hm_effect_allele   hm_beta  ...  multiancestry_hetqtest  multiancestry_df_hetqtest  multiancestry_pval_hetqtest  ci_upper  effect_allele_frequency  odds_ratio ci_lower\r
0           10_48232_G_A  rs12218882      10.0      48232.0               G                A  0.018934  ...                  62.971                         53                     0.164169       NaN                      NaN         NaN       NA\r
1           10_48486_C_T  rs10904045      10.0      48486.0               C                T -0.008312  ...                  40.767                         57                     0.948513       NaN                      NaN         NaN       NA\r
2           10_52541_A_C  rs12255619      10.0      52541.0               A                C  0.024272  ...                  66.317                         56                     0.162838       NaN                      NaN         NaN       NA\r
3           10_66015_A_G   rs7909677      10.0      66015.0               A                G  0.029994  ...                  63.938                         59                     0.307315       NaN                      NaN         NaN       NA\r
4           10_67284_T_C  rs11253113      10.0      67284.0               T                C  0.016352  ...                  49.715                         56                     0.710218       NaN                      NaN         NaN       NA\r
...                  ...         ...       ...          ...             ...              ...       ...  ...                     ...                        ...                          ...       ...                      ...         ...        ...
2001251  9_138117072_C_T   rs2606356       9.0  138117072.0               C                T  0.025101  ...                  50.285                         50                     0.462092       NaN                      NaN         NaN       NA\r
2001252  9_138117129_C_T   rs2606357       9.0  138117129.0               C                T  0.002005  ...                  63.512                         57                     0.257848       NaN                      NaN         NaN       NA\r
2001253  9_138117319_A_G   rs2606358       9.0  138117319.0               A                G -0.030354  ...                  48.075                         49                     0.510594       NaN                      NaN         NaN       NA\r
2001254  9_138117612_A_G   rs2739260       9.0  138117612.0               A                G  0.010213  ...                  70.655                         61                     0.186340       NaN                      NaN         NaN       NA\r
2001255  9_138133487_G_A  rs11137379       9.0  138133487.0               G                A  0.013703  ...                  60.324                         56                     0.322349       NaN                      NaN         NaN       NA\r

[2001256 rows x 30 columns]
'''

#Drop hm_variant_id NA
#no hm_odds_ratio that is not NaN
#DROP hm_beta NA
#NOT DROPPING hm_beta due to ... all of them are NaN
demenais_asthma.dropna(axis=0, subset=['hm_variant_id', 'hm_beta'], inplace=True)

'''
demenais_asthma
           hm_variant_id     hm_rsid  hm_chrom       hm_pos hm_other_allele hm_effect_allele   hm_beta  ...  multiancestry_hetqtest  multiancestry_df_hetqtest  multiancestry_pval_hetqtest  ci_upper  effect_allele_frequency  odds_ratio ci_lower\r
0           10_48232_G_A  rs12218882      10.0      48232.0               G                A  0.018934  ...                  62.971                         53                     0.164169       NaN                      NaN         NaN       NA\r
1           10_48486_C_T  rs10904045      10.0      48486.0               C                T -0.008312  ...                  40.767                         57                     0.948513       NaN                      NaN         NaN       NA\r
2           10_52541_A_C  rs12255619      10.0      52541.0               A                C  0.024272  ...                  66.317                         56                     0.162838       NaN                      NaN         NaN       NA\r
3           10_66015_A_G   rs7909677      10.0      66015.0               A                G  0.029994  ...                  63.938                         59                     0.307315       NaN                      NaN         NaN       NA\r
4           10_67284_T_C  rs11253113      10.0      67284.0               T                C  0.016352  ...                  49.715                         56                     0.710218       NaN                      NaN         NaN       NA\r
...                  ...         ...       ...          ...             ...              ...       ...  ...                     ...                        ...                          ...       ...                      ...         ...        ...
2001251  9_138117072_C_T   rs2606356       9.0  138117072.0               C                T  0.025101  ...                  50.285                         50                     0.462092       NaN                      NaN         NaN       NA\r
2001252  9_138117129_C_T   rs2606357       9.0  138117129.0               C                T  0.002005  ...                  63.512                         57                     0.257848       NaN                      NaN         NaN       NA\r
2001253  9_138117319_A_G   rs2606358       9.0  138117319.0               A                G -0.030354  ...                  48.075                         49                     0.510594       NaN                      NaN         NaN       NA\r
2001254  9_138117612_A_G   rs2739260       9.0  138117612.0               A                G  0.010213  ...                  70.655                         61                     0.186340       NaN                      NaN         NaN       NA\r
2001255  9_138133487_G_A  rs11137379       9.0  138133487.0               G                A  0.013703  ...                  60.324                         56                     0.322349       NaN                      NaN         NaN       NA\r

[1999262 rows x 30 columns]
'''

#Drop invalid hm_codes
print(len(demenais_asthma.index))
1999262
hm_code_mask = demenais_asthma.hm_code.ge(14) | demenais_asthma.hm_code.le(6)
hm_code_mask.sum()
0
demenais_asthma.drop(demenais_asthma[hm_code_mask].index, inplace=True)
print(len(demenais_asthma.index))
1999262

'''
demenais_asthma
           hm_variant_id     hm_rsid  hm_chrom       hm_pos hm_other_allele hm_effect_allele   hm_beta  ...  multiancestry_hetqtest  multiancestry_df_hetqtest  multiancestry_pval_hetqtest  ci_upper  effect_allele_frequency  odds_ratio ci_lower\r
0           10_48232_G_A  rs12218882      10.0      48232.0               G                A  0.018934  ...                  62.971                         53                     0.164169       NaN                      NaN         NaN       NA\r
1           10_48486_C_T  rs10904045      10.0      48486.0               C                T -0.008312  ...                  40.767                         57                     0.948513       NaN                      NaN         NaN       NA\r
2           10_52541_A_C  rs12255619      10.0      52541.0               A                C  0.024272  ...                  66.317                         56                     0.162838       NaN                      NaN         NaN       NA\r
3           10_66015_A_G   rs7909677      10.0      66015.0               A                G  0.029994  ...                  63.938                         59                     0.307315       NaN                      NaN         NaN       NA\r
4           10_67284_T_C  rs11253113      10.0      67284.0               T                C  0.016352  ...                  49.715                         56                     0.710218       NaN                      NaN         NaN       NA\r
...                  ...         ...       ...          ...             ...              ...       ...  ...                     ...                        ...                          ...       ...                      ...         ...        ...
2001251  9_138117072_C_T   rs2606356       9.0  138117072.0               C                T  0.025101  ...                  50.285                         50                     0.462092       NaN                      NaN         NaN       NA\r
2001252  9_138117129_C_T   rs2606357       9.0  138117129.0               C                T  0.002005  ...                  63.512                         57                     0.257848       NaN                      NaN         NaN       NA\r
2001253  9_138117319_A_G   rs2606358       9.0  138117319.0               A                G -0.030354  ...                  48.075                         49                     0.510594       NaN                      NaN         NaN       NA\r
2001254  9_138117612_A_G   rs2739260       9.0  138117612.0               A                G  0.010213  ...                  70.655                         61                     0.186340       NaN                      NaN         NaN       NA\r
2001255  9_138133487_G_A  rs11137379       9.0  138133487.0               G                A  0.013703  ...                  60.324                         56                     0.322349       NaN                      NaN         NaN       NA\r

[1999262 rows x 30 columns]
'''

#need to type cast, etc.
remap_dict = {'X':23}
demenais_asthma['hm_chrom'] = demenais_asthma['hm_chrom'].replace(remap_dict)
demenais_asthma['hm_chrom'] = demenais_asthma['hm_chrom'].astype('int64')
demenais_asthma['hm_pos'] = demenais_asthma['hm_pos'].astype('int64')
demenais_asthma = demenais_asthma[demenais_columns].sort_values(by=['hm_chrom', 'hm_pos'])

'''
demenais_asthma
          hm_variant_id    hm_rsid  hm_chrom    hm_pos hm_other_allele hm_effect_allele   hm_beta  hm_odds_ratio  hm_ci_lower  hm_ci_upper   p_value
703284     1_817186_G_A  rs3094315         1    817186               G                A -0.008309            NaN          NaN          NaN  0.680173
703285     1_843942_A_G  rs4040617         1    843942               A                G -0.002831            NaN          NaN          NaN  0.908569
703286     1_849670_G_A  rs2905062         1    849670               G                A  0.003518            NaN          NaN          NaN  0.901650
703287     1_850609_T_C  rs2980300         1    850609               T                C  0.003523            NaN          NaN          NaN  0.900844
703288    1_1068249_C_T  rs4075116         1   1068249               C                T  0.002322            NaN          NaN          NaN  0.883253
...                 ...        ...       ...       ...             ...              ...       ...            ...          ...          ...       ...
958866  22_50724710_C_T   rs715586        22  50724710               C                T -0.027325            NaN          NaN          NaN  0.128769
958867  22_50727236_G_A  rs8137951        22  50727236               G                A  0.006212            NaN          NaN          NaN  0.635651
958868  22_50733265_G_A   rs756638        22  50733265               G                A -0.028132            NaN          NaN          NaN  0.057367
958869  22_50737198_A_G  rs3810648        22  50737198               A                G -0.010896            NaN          NaN          NaN  0.656176
958870  22_50739662_G_A  rs2285395        22  50739662               G                A -0.010166            NaN          NaN          NaN  0.710926

[1999262 rows x 11 columns]
'''
demenais_asthma.equals(demenais_asthma.sort_values(by=['hm_chrom', 'hm_pos']))
oc_demenais = ['hm_chrom', 'hm_pos','hm_other_allele', 'hm_effect_allele', 'hm_beta', 'p_value']

'''
demenais_asthma[oc_demenais].to_csv('demenais_asthma_gwas_lz'+version+'.txt', sep='\t', index=False)
os.system("gzip -k " + 'demenais_asthma_gwas_lz'+version+'.txt &')
'''

'''
Study #8: Onengut, Type 1 Diabetes
http://ftp.ebi.ac.uk/pub/databases/gwas/summary_statistics/GCST005001-GCST006000/GCST005536/harmonised/25751624-GCST005536-EFO_0001359.h.tsv.gz
25751624
'''
Onengut = sumstatspath + "25751624-GCST005536-EFO_0001359.h.tsv"
onengut_t1d = pd.read_csv(Onengut, lineterminator='\n', sep='\t')
onengut_t1d.columns.to_list()

onengut_t1d.isna().sum()
pd.unique(onengut_t1d.hm_code)

'''
onengut_t1d.isna().sum()
hm_variant_id                  19402
hm_rsid                        19402
hm_chrom                       19402
hm_pos                         19402
hm_other_allele                19402
hm_effect_allele               19402
hm_beta                        20518
hm_odds_ratio                  19402
hm_ci_lower                    20518
hm_ci_upper                    20518
hm_effect_allele_frequency    121619
hm_code                            0
chromosome                         0
base_pair_location                 0
variant_id                         0
other_allele                    5408
effect_allele                   5408
p_value                            0
beta                               0
standard_error                  1352
odds_ratio                         0
ci_lower                        1352
ci_upper                        1352
effect_allele_frequency\r          0
dtype: int64
>>> pd.unique(onengut_t1d.hm_code)
array([10, 11,  9, 12, 13, 14, 15])
'''

onengut_columns = ['hm_variant_id', 'hm_rsid', 'hm_chrom', 'hm_pos', 'hm_other_allele', 'hm_effect_allele', 'hm_beta', 'hm_odds_ratio', 'hm_ci_lower', 'hm_ci_upper', 'p_value', 'standard_error']

'''
onengut_t1d
          hm_variant_id     hm_rsid  hm_chrom       hm_pos hm_other_allele hm_effect_allele   hm_beta  hm_odds_ratio  ...  effect_allele   p_value      beta  standard_error  odds_ratio  ci_lower  ci_upper effect_allele_frequency\r
0         10_251693_A_G   rs3125037      10.0     251693.0               A                G -0.006018       0.994000  ...              G  0.775097 -0.006018        0.021063      0.9940  0.953799  1.035895                      NA\r
1         10_308361_C_T  rs35198327      10.0     308361.0               C                T -0.001998       0.998004  ...              C  0.916167  0.001998        0.018981      1.0020  0.965408  1.039979                      NA\r
2         10_318459_A_G   rs9804310      10.0     318459.0               A                G -0.046044       0.955000  ...              G  0.076855 -0.046044        0.026025      0.9550  0.907508  1.004977                      NA\r
3         10_361644_G_A   rs3922851      10.0     361644.0               G                A  0.011039       1.011100  ...              A  0.562625  0.011039        0.019067      1.0111  0.974011  1.049601                      NA\r
4         10_475246_C_T  rs11252630      10.0     475246.0               C                T  0.008960       1.009000  ...              T  0.596520  0.008960        0.016924      1.0090  0.976080  1.043031                      NA\r
...                 ...         ...       ...          ...             ...              ...       ...            ...  ...            ...       ...       ...             ...         ...       ...       ...                       ...
121614  9_137245305_G_A   rs9775264       9.0  137245305.0               G                A -0.085994       0.917600  ...              G  0.333898  0.085994        0.088994      1.0898  0.915363  1.297478                      NA\r
121615  9_137397708_G_A  rs11523300       9.0  137397708.0               G                A -0.021019       0.979200  ...              A  0.339808 -0.021019        0.022020      0.9792  0.937837  1.022387                      NA\r
121616  9_137422632_C_A  rs13288824       9.0  137422632.0               C                A  0.041046       1.041900  ...              A  0.030936  0.041046        0.019021      1.0419  1.003771  1.081477                      NA\r
121617  9_137735488_C_T   rs9410127       9.0  137735488.0               C                T  0.075015       1.077900  ...              T  0.118170  0.075015        0.048009      1.0779  0.981097  1.184254                      NA\r
121618  9_138096907_A_G   rs3812541       9.0  138096907.0               A                G -0.003005       0.997000  ...              G  0.896222 -0.003005        0.023035      0.9970  0.952989  1.043044                      NA\r

[121619 rows x 24 columns]
'''

#Drop hm_variant_id NA
#no hm_odds_ratio that is not NaN
#DROP hm_beta NA
#drop hm_odds_ratio due to ... all of them are NaN
onengut_t1d.dropna(axis=0, subset=['hm_variant_id', 'hm_beta', 'hm_odds_ratio'], inplace=True)

'''
onengut_t1d
          hm_variant_id     hm_rsid  hm_chrom       hm_pos hm_other_allele hm_effect_allele   hm_beta  hm_odds_ratio  ...  effect_allele   p_value      beta  standard_error  odds_ratio  ci_lower  ci_upper effect_allele_frequency\r
0         10_251693_A_G   rs3125037      10.0     251693.0               A                G -0.006018       0.994000  ...              G  0.775097 -0.006018        0.021063      0.9940  0.953799  1.035895                      NA\r
1         10_308361_C_T  rs35198327      10.0     308361.0               C                T -0.001998       0.998004  ...              C  0.916167  0.001998        0.018981      1.0020  0.965408  1.039979                      NA\r
2         10_318459_A_G   rs9804310      10.0     318459.0               A                G -0.046044       0.955000  ...              G  0.076855 -0.046044        0.026025      0.9550  0.907508  1.004977                      NA\r
3         10_361644_G_A   rs3922851      10.0     361644.0               G                A  0.011039       1.011100  ...              A  0.562625  0.011039        0.019067      1.0111  0.974011  1.049601                      NA\r
4         10_475246_C_T  rs11252630      10.0     475246.0               C                T  0.008960       1.009000  ...              T  0.596520  0.008960        0.016924      1.0090  0.976080  1.043031                      NA\r
...                 ...         ...       ...          ...             ...              ...       ...            ...  ...            ...       ...       ...             ...         ...       ...       ...                       ...
121614  9_137245305_G_A   rs9775264       9.0  137245305.0               G                A -0.085994       0.917600  ...              G  0.333898  0.085994        0.088994      1.0898  0.915363  1.297478                      NA\r
121615  9_137397708_G_A  rs11523300       9.0  137397708.0               G                A -0.021019       0.979200  ...              A  0.339808 -0.021019        0.022020      0.9792  0.937837  1.022387                      NA\r
121616  9_137422632_C_A  rs13288824       9.0  137422632.0               C                A  0.041046       1.041900  ...              A  0.030936  0.041046        0.019021      1.0419  1.003771  1.081477                      NA\r
121617  9_137735488_C_T   rs9410127       9.0  137735488.0               C                T  0.075015       1.077900  ...              T  0.118170  0.075015        0.048009      1.0779  0.981097  1.184254                      NA\r
121618  9_138096907_A_G   rs3812541       9.0  138096907.0               A                G -0.003005       0.997000  ...              G  0.896222 -0.003005        0.023035      0.9970  0.952989  1.043044                      NA\r

[101101 rows x 24 columns]
'''

#Drop invalid hm_codes
print(len(onengut_t1d.index))
101101
hm_code_mask = onengut_t1d.hm_code.ge(14) | onengut_t1d.hm_code.le(6)
hm_code_mask.sum()
0
onengut_t1d.drop(onengut_t1d[hm_code_mask].index, inplace=True)
print(len(onengut_t1d.index))
101101

'''
onengut_t1d
          hm_variant_id     hm_rsid  hm_chrom       hm_pos hm_other_allele hm_effect_allele   hm_beta  hm_odds_ratio  ...  effect_allele   p_value      beta  standard_error  odds_ratio  ci_lower  ci_upper effect_allele_frequency\r
0         10_251693_A_G   rs3125037      10.0     251693.0               A                G -0.006018       0.994000  ...              G  0.775097 -0.006018        0.021063      0.9940  0.953799  1.035895                      NA\r
1         10_308361_C_T  rs35198327      10.0     308361.0               C                T -0.001998       0.998004  ...              C  0.916167  0.001998        0.018981      1.0020  0.965408  1.039979                      NA\r
2         10_318459_A_G   rs9804310      10.0     318459.0               A                G -0.046044       0.955000  ...              G  0.076855 -0.046044        0.026025      0.9550  0.907508  1.004977                      NA\r
3         10_361644_G_A   rs3922851      10.0     361644.0               G                A  0.011039       1.011100  ...              A  0.562625  0.011039        0.019067      1.0111  0.974011  1.049601                      NA\r
4         10_475246_C_T  rs11252630      10.0     475246.0               C                T  0.008960       1.009000  ...              T  0.596520  0.008960        0.016924      1.0090  0.976080  1.043031                      NA\r
...                 ...         ...       ...          ...             ...              ...       ...            ...  ...            ...       ...       ...             ...         ...       ...       ...                       ...
121614  9_137245305_G_A   rs9775264       9.0  137245305.0               G                A -0.085994       0.917600  ...              G  0.333898  0.085994        0.088994      1.0898  0.915363  1.297478                      NA\r
121615  9_137397708_G_A  rs11523300       9.0  137397708.0               G                A -0.021019       0.979200  ...              A  0.339808 -0.021019        0.022020      0.9792  0.937837  1.022387                      NA\r
121616  9_137422632_C_A  rs13288824       9.0  137422632.0               C                A  0.041046       1.041900  ...              A  0.030936  0.041046        0.019021      1.0419  1.003771  1.081477                      NA\r
121617  9_137735488_C_T   rs9410127       9.0  137735488.0               C                T  0.075015       1.077900  ...              T  0.118170  0.075015        0.048009      1.0779  0.981097  1.184254                      NA\r
121618  9_138096907_A_G   rs3812541       9.0  138096907.0               A                G -0.003005       0.997000  ...              G  0.896222 -0.003005        0.023035      0.9970  0.952989  1.043044                      NA\r

[101101 rows x 24 columns]
'''

#need to type cast, etc.
remap_dict = {'X':23}
onengut_t1d['hm_chrom'] = onengut_t1d['hm_chrom'].replace(remap_dict)
onengut_t1d['hm_chrom'] = onengut_t1d['hm_chrom'].astype('int64')
onengut_t1d['hm_pos'] = onengut_t1d['hm_pos'].astype('int64')
onengut_t1d = onengut_t1d[onengut_columns].sort_values(by=['hm_chrom', 'hm_pos'])

'''
onengut_t1d
         hm_variant_id     hm_rsid  hm_chrom    hm_pos hm_other_allele hm_effect_allele   hm_beta  hm_odds_ratio  hm_ci_lower  hm_ci_upper   p_value
41386    1_1182895_C_T  rs61733845         1   1182895               C                T -0.073001         0.9296     0.851120     1.015316  0.104756
41387    1_1199862_A_C   rs9729550         1   1199862               A                C -0.038949         0.9618     0.923066     1.002159  0.063291
41388    1_1205055_G_T   rs1815606         1   1205055               G                T -0.027988         0.9724     0.935035     1.011258  0.161513
41389    1_1228424_C_T   rs7515488         1   1228424               C                T -0.050031         0.9512     0.903913     1.000961  0.054470
41390    1_1229930_G_A  rs11260562         1   1229930               G                A -0.005013         0.9950     0.921600     1.074246  0.897987
...                ...         ...       ...       ...             ...              ...       ...            ...          ...          ...       ...
61177  22_50639823_C_T   rs4040041        22  50639823               C                T  0.005982         1.0060     0.969333     1.044054  0.752162
61178  22_50656498_C_T   rs9616810        22  50656498               C                T  0.034981         1.0356     0.991917     1.081206  0.111630
61179  22_50667128_C_T   rs9616812        22  50667128               C                T -0.003005         0.9970     0.962388     1.032857  0.867632
61180  22_50671564_T_C   rs9628185        22  50671564               T                C -0.001001         0.9990     0.964353     1.034892  0.955696
61181  22_50718238_C_T   rs9628187        22  50718238               C                T -0.017044         0.9831     0.939655     1.028553  0.459828

[101101 rows x 11 columns]
'''
onengut_t1d.equals(onengut_t1d.sort_values(by=['hm_chrom', 'hm_pos']))
oc_onengut = ['hm_chrom', 'hm_pos','hm_other_allele', 'hm_effect_allele', 'hm_beta', 'p_value']

oc_metal = ['hm_variant_id', 'hm_other_allele', 'hm_effect_allele', 'hm_beta', 'standard_error', 'p_value']
onengut_t1d[oc_metal].to_csv('onengut_t1d_gwas_metal-'+version+'.txt', sep='\t', index=False)

'''
#locus zoom setup
onengut_t1d[oc_onengut].to_csv('onengut_t1d_gwas_lz'+version+'.txt', sep='\t', index=False)
os.system("gzip -k " + 'onengut_t1d_gwas_lz'+version+'.txt &')
'''

'''
Study #9: Sliz Atopic dermatitis
http://ftp.ebi.ac.uk/pub/databases/gwas/summary_statistics/GCST90027001-GCST90028000/GCST90027161/harmonised/34454985-GCST90027161-EFO_0000274.h.tsv.gz
34454985
'''
Sliz = sumstatspath + "34454985-GCST90027161-EFO_0000274.h.tsv"
sliz_ad = pd.read_csv(Sliz, lineterminator='\n', sep='\t')
sliz_ad.columns.to_list()

sliz_ad.isna().sum()
pd.unique(sliz_ad.hm_code)

'''
sliz_ad.isna().sum()
hm_variant_id                    28729
hm_rsid                          28729
hm_chrom                         28729
hm_pos                           28729
hm_other_allele                  28729
hm_effect_allele                 28729
hm_beta                          52334
hm_odds_ratio                 16242371
hm_ci_lower                   16242371
hm_ci_upper                   16242371
hm_effect_allele_frequency       57516
hm_code                              0
variant_id                           0
p_value                              0
chromosome                           0
base_pair_location                   0
effect_allele                        0
other_allele                         0
beta                                 0
standard_error                       0
effect_allele_frequency             24
odds_ratio                    16242371
ci_lower                      16242371
ci_upper\r                           0
dtype: int64
>>> pd.unique(sliz_ad.hm_code)
array([10,  5, 11,  6, 15, 16])
'''

sliz_columns = ['hm_variant_id', 'hm_rsid', 'hm_chrom', 'hm_pos', 'hm_other_allele', 'hm_effect_allele', 'hm_beta', 'hm_odds_ratio', 'hm_ci_lower', 'hm_ci_upper', 'p_value', 'standard_error']

'''
sliz_ad
            hm_variant_id       hm_rsid hm_chrom       hm_pos hm_other_allele hm_effect_allele  hm_beta  hm_odds_ratio  ...  effect_allele  other_allele    beta  standard_error effect_allele_frequency  odds_ratio  ci_lower  ci_upper\r
0            1_115637_G_A    rs74337086        1     115637.0               G                A   0.4986            NaN  ...              A             G  0.4986          0.2091                 0.00080         NaN       NaN        NA\r
1            1_137928_G_A  rs1292537993        1     137928.0               G                A  -0.6610            NaN  ...              A             G -0.6610          0.3785                 0.00020         NaN       NaN        NA\r
2            1_184280_C_T  rs1264891165        1     184280.0               C                T  -0.1474            NaN  ...              T             C -0.1474          0.2624                 0.00050         NaN       NaN        NA\r
3            1_502543_G_C  rs1179030655        1     502543.0               G                C  -0.0515            NaN  ...              C             G -0.0515          0.2037                 0.00070         NaN       NaN        NA\r
4            1_588427_G_C  rs1223421187        1     588427.0               G                C   0.5031            NaN  ...              C             G  0.5031          0.7032                 0.00007         NaN       NaN        NA\r
...                   ...           ...      ...          ...             ...              ...      ...            ...  ...            ...           ...     ...             ...                     ...         ...       ...         ...
16242366  X_155697182_T_C   rs192521690        X  155697182.0               T                C   0.1248            NaN  ...              T             C -0.1248          0.0876                 0.99760         NaN       NaN        NA\r
16242367  X_155697920_G_A      rs644138        X  155697920.0               G                A  -0.0222            NaN  ...              A             G -0.0222          0.0134                 0.24720         NaN       NaN        NA\r
16242368  X_155698443_C_A   rs186130524        X  155698443.0               C                A  -0.0563            NaN  ...              A             C -0.0563          0.1370                 0.00100         NaN       NaN        NA\r
16242369  X_155698490_C_T   rs144607509        X  155698490.0               C                T   0.0282            NaN  ...              T             C  0.0282          0.0511                 0.00750         NaN       NaN        NA\r
16242370  X_155699751_C_T      rs557132        X  155699751.0               C                T  -0.0205            NaN  ...              T             C -0.0205          0.0142                 0.12820         NaN       NaN        NA\r

[16242371 rows x 24 columns]
'''

#Drop hm_variant_id NA
#no hm_odds_ratio that is not NaN
#DROP hm_beta NA
#do not drop hm_odds_ratio due to ... all of them are NaN
sliz_ad.dropna(axis=0, subset=['hm_variant_id', 'hm_beta'], inplace=True)

'''
sliz_ad
            hm_variant_id       hm_rsid hm_chrom       hm_pos hm_other_allele hm_effect_allele  hm_beta  hm_odds_ratio  ...  effect_allele  other_allele    beta  standard_error effect_allele_frequency  odds_ratio  ci_lower  ci_upper\r
0            1_115637_G_A    rs74337086        1     115637.0               G                A   0.4986            NaN  ...              A             G  0.4986          0.2091                 0.00080         NaN       NaN        NA\r
1            1_137928_G_A  rs1292537993        1     137928.0               G                A  -0.6610            NaN  ...              A             G -0.6610          0.3785                 0.00020         NaN       NaN        NA\r
2            1_184280_C_T  rs1264891165        1     184280.0               C                T  -0.1474            NaN  ...              T             C -0.1474          0.2624                 0.00050         NaN       NaN        NA\r
3            1_502543_G_C  rs1179030655        1     502543.0               G                C  -0.0515            NaN  ...              C             G -0.0515          0.2037                 0.00070         NaN       NaN        NA\r
4            1_588427_G_C  rs1223421187        1     588427.0               G                C   0.5031            NaN  ...              C             G  0.5031          0.7032                 0.00007         NaN       NaN        NA\r
...                   ...           ...      ...          ...             ...              ...      ...            ...  ...            ...           ...     ...             ...                     ...         ...       ...         ...
16242366  X_155697182_T_C   rs192521690        X  155697182.0               T                C   0.1248            NaN  ...              T             C -0.1248          0.0876                 0.99760         NaN       NaN        NA\r
16242367  X_155697920_G_A      rs644138        X  155697920.0               G                A  -0.0222            NaN  ...              A             G -0.0222          0.0134                 0.24720         NaN       NaN        NA\r
16242368  X_155698443_C_A   rs186130524        X  155698443.0               C                A  -0.0563            NaN  ...              A             C -0.0563          0.1370                 0.00100         NaN       NaN        NA\r
16242369  X_155698490_C_T   rs144607509        X  155698490.0               C                T   0.0282            NaN  ...              T             C  0.0282          0.0511                 0.00750         NaN       NaN        NA\r
16242370  X_155699751_C_T      rs557132        X  155699751.0               C                T  -0.0205            NaN  ...              T             C -0.0205          0.0142                 0.12820         NaN       NaN        NA\r

[16190037 rows x 24 columns]
'''

#Drop invalid hm_codes
print(len(sliz_ad.index))
16190037
hm_code_mask = sliz_ad.hm_code.ge(14) | sliz_ad.hm_code.le(6)
hm_code_mask.sum()
2305956
sliz_ad.drop(sliz_ad[hm_code_mask].index, inplace=True)
print(len(sliz_ad.index))
13884081

'''
sliz_ad
            hm_variant_id       hm_rsid hm_chrom       hm_pos hm_other_allele hm_effect_allele  hm_beta  hm_odds_ratio  ...  effect_allele  other_allele    beta  standard_error effect_allele_frequency  odds_ratio  ci_lower  ci_upper\r
0            1_115637_G_A    rs74337086        1     115637.0               G                A   0.4986            NaN  ...              A             G  0.4986          0.2091                  0.0008         NaN       NaN        NA\r
1            1_137928_G_A  rs1292537993        1     137928.0               G                A  -0.6610            NaN  ...              A             G -0.6610          0.3785                  0.0002         NaN       NaN        NA\r
2            1_184280_C_T  rs1264891165        1     184280.0               C                T  -0.1474            NaN  ...              T             C -0.1474          0.2624                  0.0005         NaN       NaN        NA\r
5            1_591353_C_T    rs28760963        1     591353.0               C                T   0.1566            NaN  ...              T             C  0.1566          0.1723                  0.0012         NaN       NaN        NA\r
6            1_593262_G_A    rs76388980        1     593262.0               G                A  -0.1636            NaN  ...              A             G -0.1636          0.1829                  0.0010         NaN       NaN        NA\r
...                   ...           ...      ...          ...             ...              ...      ...            ...  ...            ...           ...     ...             ...                     ...         ...       ...         ...
16242366  X_155697182_T_C   rs192521690        X  155697182.0               T                C   0.1248            NaN  ...              T             C -0.1248          0.0876                  0.9976         NaN       NaN        NA\r
16242367  X_155697920_G_A      rs644138        X  155697920.0               G                A  -0.0222            NaN  ...              A             G -0.0222          0.0134                  0.2472         NaN       NaN        NA\r
16242368  X_155698443_C_A   rs186130524        X  155698443.0               C                A  -0.0563            NaN  ...              A             C -0.0563          0.1370                  0.0010         NaN       NaN        NA\r
16242369  X_155698490_C_T   rs144607509        X  155698490.0               C                T   0.0282            NaN  ...              T             C  0.0282          0.0511                  0.0075         NaN       NaN        NA\r
16242370  X_155699751_C_T      rs557132        X  155699751.0               C                T  -0.0205            NaN  ...              T             C -0.0205          0.0142                  0.1282         NaN       NaN        NA\r

[13884081 rows x 24 columns]
'''

#need to type cast, etc.
remap_dict = {'X':23}
sliz_ad['hm_chrom'] = sliz_ad['hm_chrom'].replace(remap_dict)
sliz_ad['hm_chrom'] = sliz_ad['hm_chrom'].astype('int64')
sliz_ad['hm_pos'] = sliz_ad['hm_pos'].astype('int64')
sliz_ad = sliz_ad[sliz_columns].sort_values(by=['hm_chrom', 'hm_pos'])

'''
sliz_ad
            hm_variant_id       hm_rsid  hm_chrom     hm_pos hm_other_allele hm_effect_allele  hm_beta  hm_odds_ratio  hm_ci_lower  hm_ci_upper  p_value
0            1_115637_G_A    rs74337086         1     115637               G                A   0.4986            NaN          NaN          NaN  0.01710
1            1_137928_G_A  rs1292537993         1     137928               G                A  -0.6610            NaN          NaN          NaN  0.08077
2            1_184280_C_T  rs1264891165         1     184280               C                T  -0.1474            NaN          NaN          NaN  0.57440
5            1_591353_C_T    rs28760963         1     591353               C                T   0.1566            NaN          NaN          NaN  0.36330
6            1_593262_G_A    rs76388980         1     593262               G                A  -0.1636            NaN          NaN          NaN  0.37090
...                   ...           ...       ...        ...             ...              ...      ...            ...          ...          ...      ...
16242366  X_155697182_T_C   rs192521690        23  155697182               T                C   0.1248            NaN          NaN          NaN  0.15440
16242367  X_155697920_G_A      rs644138        23  155697920               G                A  -0.0222            NaN          NaN          NaN  0.09712
16242368  X_155698443_C_A   rs186130524        23  155698443               C                A  -0.0563            NaN          NaN          NaN  0.68080
16242369  X_155698490_C_T   rs144607509        23  155698490               C                T   0.0282            NaN          NaN          NaN  0.58020
16242370  X_155699751_C_T      rs557132        23  155699751               C                T  -0.0205            NaN          NaN          NaN  0.14670

[13884081 rows x 11 columns]
'''
sliz_ad.equals(sliz_ad.sort_values(by=['hm_chrom', 'hm_pos']))
oc_sliz = ['hm_chrom', 'hm_pos','hm_other_allele', 'hm_effect_allele', 'hm_beta', 'p_value']

'''
sliz_ad[oc_sliz].to_csv('sliz_ad_gwas_lz'+version+'.txt', sep='\t', index=False)
os.system("gzip -k " + 'sliz_ad_gwas_lz'+version+'.txt &')
'''

'''
Study #X: Grosche Eczema
– Excluded since there are no betas or odds ratios
http://ftp.ebi.ac.uk/pub/databases/gwas/summary_statistics/GCST90044001-GCST90045000/GCST90044763/harmonised/34785669-GCST90044763-HP_0000964.h.tsv.gz
34785669
'''
'''
Grosche = sumstatspath + "34785669-GCST90044763-HP_0000964.h.tsv"
grosche_eczema = pd.read_csv(Grosche, lineterminator='\n', sep='\t')
grosche_eczema.columns.to_list()

grosche_eczema.isna().sum()
pd.unique(grosche_eczema.hm_code)
'''
'''
grosche_eczema.isna().sum()
hm_variant_id                   14428
hm_rsid                         14428
hm_chrom                        14428
hm_pos                          14428
hm_other_allele                 14428
hm_effect_allele                14428
hm_beta                       9397031
hm_odds_ratio                 9397031
hm_ci_lower                   9397031
hm_ci_upper                   9397031
hm_effect_allele_frequency      14428
hm_code                             0
variant_id                          0
chromosome                          0
base_pair_location                  0
effect_allele                       0
other_allele                        0
effect_allele_frequency             0
p_value                             0
odds_ratio                    9397031
ci_lower                      9397031
ci_upper                      9397031
beta                          9397031
standard_error\r                    0
dtype: int64
>>> pd.unique(grosche_eczema.hm_code)
array([10, 11,  6,  5, 15])
'''

#grosche_columns = ['hm_variant_id', 'hm_rsid', 'hm_chrom', 'hm_pos', 'hm_other_allele', 'hm_effect_allele', 'hm_beta', 'hm_odds_ratio', 'hm_ci_lower', 'hm_ci_upper', 'p_value', 'standard_error']

'''
grosche_eczema
           hm_variant_id      hm_rsid  hm_chrom       hm_pos hm_other_allele hm_effect_allele  hm_beta  hm_odds_ratio  ...  other_allele  effect_allele_frequency        p_value  odds_ratio ci_lower  ci_upper  beta standard_error\r
0        1_152313385_G_A   rs61816761       1.0  152313385.0               G                A      NaN            NaN  ...             g                   0.0176  3.740000e-118         NaN      NaN       NaN   NaN             NA\r
1        1_152347096_T_C   rs61816766       1.0  152347096.0               T                C      NaN            NaN  ...             c                   0.9744   6.310000e-92         NaN      NaN       NaN   NaN             NA\r
2        1_152298743_A_T   rs61815559       1.0  152298743.0               A                T      NaN            NaN  ...             t                   0.9736   9.890000e-90         NaN      NaN       NaN   NaN             NA\r
3        1_152206676_C_T   rs12123821       1.0  152206676.0               C                T      NaN            NaN  ...             c                   0.0448   4.020000e-85         NaN      NaN       NaN   NaN             NA\r
4        1_152027641_G_A  rs115288876       1.0  152027641.0               G                A      NaN            NaN  ...             g                   0.0420   3.170000e-81         NaN      NaN       NaN   NaN             NA\r
...                  ...          ...       ...          ...             ...              ...      ...            ...  ...           ...                      ...            ...         ...      ...       ...   ...              ...
9397026              NaN          NaN       NaN          NaN             NaN              NaN      NaN            NaN  ...             c                   0.6591   9.897000e-01         NaN      NaN       NaN   NaN             NA\r
9397027              NaN          NaN       NaN          NaN             NaN              NaN      NaN            NaN  ...             c                   0.1007   9.907000e-01         NaN      NaN       NaN   NaN             NA\r
9397028              NaN          NaN       NaN          NaN             NaN              NaN      NaN            NaN  ...             g                   0.9760   9.944000e-01         NaN      NaN       NaN   NaN             NA\r
9397029              NaN          NaN       NaN          NaN             NaN              NaN      NaN            NaN  ...             g                   0.0604   9.965000e-01         NaN      NaN       NaN   NaN             NA\r
9397030              NaN          NaN       NaN          NaN             NaN              NaN      NaN            NaN  ...             g                   0.7615   9.974000e-01         NaN      NaN       NaN   NaN             NA\r

[9397031 rows x 24 columns]
'''


'''
Study #11: Dubois Celiac Disease
http://ftp.ebi.ac.uk/pub/databases/gwas/summary_statistics/GCST000001-GCST001000/GCST000612/harmonised/20190752-GCST000612-EFO_0001060.h.tsv.gz
20190752
'''
Dubois = sumstatspath + "20190752-GCST000612-EFO_0001060.h.tsv"
dubois_cd = pd.read_csv(Dubois, lineterminator='\n', sep='\t')
dubois_cd.columns.to_list()

dubois_cd.isna().sum()
pd.unique(dubois_cd.hm_code)

'''
 dubois_cd.isna().sum()
hm_variant_id                   1983
hm_rsid                         1983
hm_chrom                        1983
hm_pos                          1983
hm_other_allele                 1983
hm_effect_allele                1983
hm_beta                         5098
hm_odds_ratio                   1983
hm_ci_lower                     2001
hm_ci_upper                     2001
hm_effect_allele_frequency    523390
hm_code                            0
chromosome                         0
base_pair_location                 0
variant_id                         0
other_allele                       0
effect_allele                      0
p_value                            0
beta                               0
standard_error                    18
odds_ratio                         0
ci_lower                          18
ci_upper                          18
effect_allele_frequency\r          0
dtype: int64
>>> pd.unique(dubois_cd.hm_code)
array([10, 12, 11, 13,  9, 15])
'''

dubois_columns = ['hm_variant_id', 'hm_rsid', 'hm_chrom', 'hm_pos', 'hm_other_allele', 'hm_effect_allele', 'hm_beta', 'hm_odds_ratio', 'hm_ci_lower', 'hm_ci_upper', 'p_value', 'standard_error']

'''
dubois_cd
          hm_variant_id     hm_rsid hm_chrom       hm_pos hm_other_allele hm_effect_allele   hm_beta  hm_odds_ratio  ...  effect_allele  p_value      beta  standard_error  odds_ratio  ci_lower  ci_upper effect_allele_frequency\r
0          10_80130_C_T  rs11591988       10      80130.0               C                T  0.026642         1.0270  ...              T  0.54530  0.026642        0.044050      1.0270  0.942052  1.119608                      NA\r
1         10_106057_T_C   rs9286070       10     106057.0               T                C -0.055407         0.9461  ...              C  0.53190 -0.055407        0.088636      0.9461  0.795223  1.125603                      NA\r
2         10_113006_C_T  rs11253562       10     113006.0               C                T       NaN         1.0000  ...              T  0.99410  0.000000        0.000000      1.0000  1.000000  1.000000                      NA\r
3         10_113136_G_A   rs4881551       10     113136.0               G                A  0.026642         1.0270  ...              T  0.30520  0.026642        0.025983      1.0270  0.976007  1.080657                      NA\r
4         10_113464_G_A   rs4880750       10     113464.0               G                A  0.019803         1.0200  ...              A  0.52200  0.019803        0.030929      1.0200  0.960004  1.083745                      NA\r
...                 ...         ...      ...          ...             ...              ...       ...            ...  ...            ...      ...       ...             ...         ...       ...       ...                       ...
523385  X_155988327_A_C   rs5983854        X  155988327.0               A                C  0.047837         1.0490  ...              C  0.09744  0.047837        0.028863      1.0490  0.991304  1.110054                      NA\r
523386    Y_2800415_T_C   rs2058276        Y    2800415.0               T                C  0.094401         1.0990  ...              G  0.25160  0.094401        0.082340      1.0990  0.935208  1.291478                      NA\r
523387    Y_7000077_A_G   rs1865680        Y    7000077.0               A                G  0.138892         1.1490  ...              G  0.09224  0.138892        0.082492      1.1490  0.977465  1.350638                      NA\r
523388   Y_12914512_C_A   rs2032624        Y   12914512.0               C                A  0.132781         1.1420  ...              T  0.10860  0.132781        0.082755      1.1420  0.971011  1.343099                      NA\r
523389   Y_19555322_C_T   rs3848982        Y   19555322.0               C                T  0.038259         1.0390  ...              T  0.84870  0.038259        0.200541      1.0390  0.701312  1.539287                      NA\r

[523390 rows x 24 columns]
'''

#Drop hm_variant_id NA
#no hm_odds_ratio that is not NaN
#DROP hm_beta NA
#do not drop hm_odds_ratio due to ... all of them are NaN
dubois_cd.dropna(axis=0, subset=['hm_variant_id', 'hm_beta', 'hm_odds_ratio'], inplace=True)

'''
dubois_cd
          hm_variant_id     hm_rsid hm_chrom       hm_pos hm_other_allele hm_effect_allele   hm_beta  hm_odds_ratio  ...  effect_allele  p_value      beta  standard_error  odds_ratio  ci_lower  ci_upper effect_allele_frequency\r
0          10_80130_C_T  rs11591988       10      80130.0               C                T  0.026642       1.027000  ...              T  0.54530  0.026642        0.044050      1.0270  0.942052  1.119608                      NA\r
1         10_106057_T_C   rs9286070       10     106057.0               T                C -0.055407       0.946100  ...              C  0.53190 -0.055407        0.088636      0.9461  0.795223  1.125603                      NA\r
3         10_113136_G_A   rs4881551       10     113136.0               G                A  0.026642       1.027000  ...              T  0.30520  0.026642        0.025983      1.0270  0.976007  1.080657                      NA\r
4         10_113464_G_A   rs4880750       10     113464.0               G                A  0.019803       1.020000  ...              A  0.52200  0.019803        0.030929      1.0200  0.960004  1.083745                      NA\r
5         10_116162_A_C  rs11594819       10     116162.0               A                C  0.001802       1.001803  ...              A  0.94880 -0.001802        0.028057      0.9982  0.944790  1.054629                      NA\r
...                 ...         ...      ...          ...             ...              ...       ...            ...  ...            ...      ...       ...             ...         ...       ...       ...                       ...
523385  X_155988327_A_C   rs5983854        X  155988327.0               A                C  0.047837       1.049000  ...              C  0.09744  0.047837        0.028863      1.0490  0.991304  1.110054                      NA\r
523386    Y_2800415_T_C   rs2058276        Y    2800415.0               T                C  0.094401       1.099000  ...              G  0.25160  0.094401        0.082340      1.0990  0.935208  1.291478                      NA\r
523387    Y_7000077_A_G   rs1865680        Y    7000077.0               A                G  0.138892       1.149000  ...              G  0.09224  0.138892        0.082492      1.1490  0.977465  1.350638                      NA\r
523388   Y_12914512_C_A   rs2032624        Y   12914512.0               C                A  0.132781       1.142000  ...              T  0.10860  0.132781        0.082755      1.1420  0.971011  1.343099                      NA\r
523389   Y_19555322_C_T   rs3848982        Y   19555322.0               C                T  0.038259       1.039000  ...              T  0.84870  0.038259        0.200541      1.0390  0.701312  1.539287                      NA\r

[518292 rows x 24 columns]
'''

#Drop invalid hm_codes
print(len(dubois_cd.index))
518292
hm_code_mask = dubois_cd.hm_code.ge(14) | dubois_cd.hm_code.le(6)
hm_code_mask.sum()
0
dubois_cd.drop(dubois_cd[hm_code_mask].index, inplace=True)
print(len(dubois_cd.index))
518292

'''
dubois_cd
          hm_variant_id     hm_rsid hm_chrom       hm_pos hm_other_allele hm_effect_allele   hm_beta  hm_odds_ratio  ...  effect_allele  p_value      beta  standard_error  odds_ratio  ci_lower  ci_upper effect_allele_frequency\r
0          10_80130_C_T  rs11591988       10      80130.0               C                T  0.026642       1.027000  ...              T  0.54530  0.026642        0.044050      1.0270  0.942052  1.119608                      NA\r
1         10_106057_T_C   rs9286070       10     106057.0               T                C -0.055407       0.946100  ...              C  0.53190 -0.055407        0.088636      0.9461  0.795223  1.125603                      NA\r
3         10_113136_G_A   rs4881551       10     113136.0               G                A  0.026642       1.027000  ...              T  0.30520  0.026642        0.025983      1.0270  0.976007  1.080657                      NA\r
4         10_113464_G_A   rs4880750       10     113464.0               G                A  0.019803       1.020000  ...              A  0.52200  0.019803        0.030929      1.0200  0.960004  1.083745                      NA\r
5         10_116162_A_C  rs11594819       10     116162.0               A                C  0.001802       1.001803  ...              A  0.94880 -0.001802        0.028057      0.9982  0.944790  1.054629                      NA\r
...                 ...         ...      ...          ...             ...              ...       ...            ...  ...            ...      ...       ...             ...         ...       ...       ...                       ...
523385  X_155988327_A_C   rs5983854        X  155988327.0               A                C  0.047837       1.049000  ...              C  0.09744  0.047837        0.028863      1.0490  0.991304  1.110054                      NA\r
523386    Y_2800415_T_C   rs2058276        Y    2800415.0               T                C  0.094401       1.099000  ...              G  0.25160  0.094401        0.082340      1.0990  0.935208  1.291478                      NA\r
523387    Y_7000077_A_G   rs1865680        Y    7000077.0               A                G  0.138892       1.149000  ...              G  0.09224  0.138892        0.082492      1.1490  0.977465  1.350638                      NA\r
523388   Y_12914512_C_A   rs2032624        Y   12914512.0               C                A  0.132781       1.142000  ...              T  0.10860  0.132781        0.082755      1.1420  0.971011  1.343099                      NA\r
523389   Y_19555322_C_T   rs3848982        Y   19555322.0               C                T  0.038259       1.039000  ...              T  0.84870  0.038259        0.200541      1.0390  0.701312  1.539287                      NA\r

[518292 rows x 24 columns]
'''

'''
Drop Y chromosome
'''
y_chr_mask = dubois_cd.hm_chrom.eq('Y')
print(len(dubois_cd.index))
518292
dubois_cd.drop(dubois_cd[y_chr_mask].index, inplace=True)
print(len(dubois_cd.index))
518288

'''
dubois_cd
          hm_variant_id     hm_rsid hm_chrom       hm_pos hm_other_allele hm_effect_allele   hm_beta  hm_odds_ratio  ...  effect_allele  p_value      beta  standard_error  odds_ratio  ci_lower  ci_upper effect_allele_frequency\r
0          10_80130_C_T  rs11591988       10      80130.0               C                T  0.026642       1.027000  ...              T  0.54530  0.026642        0.044050      1.0270  0.942052  1.119608                      NA\r
1         10_106057_T_C   rs9286070       10     106057.0               T                C -0.055407       0.946100  ...              C  0.53190 -0.055407        0.088636      0.9461  0.795223  1.125603                      NA\r
3         10_113136_G_A   rs4881551       10     113136.0               G                A  0.026642       1.027000  ...              T  0.30520  0.026642        0.025983      1.0270  0.976007  1.080657                      NA\r
4         10_113464_G_A   rs4880750       10     113464.0               G                A  0.019803       1.020000  ...              A  0.52200  0.019803        0.030929      1.0200  0.960004  1.083745                      NA\r
5         10_116162_A_C  rs11594819       10     116162.0               A                C  0.001802       1.001803  ...              A  0.94880 -0.001802        0.028057      0.9982  0.944790  1.054629                      NA\r
...                 ...         ...      ...          ...             ...              ...       ...            ...  ...            ...      ...       ...             ...         ...       ...       ...                       ...
523380    X_2559146_A_G  rs17842890        X    2559146.0               A                G -0.023781       0.976500  ...              C  0.77820 -0.023781        0.084428      0.9765  0.827571  1.152230                      NA\r
523381    X_2559663_A_G  rs17842893        X    2559663.0               A                G  0.031904       1.032418  ...              A  0.70570 -0.031904        0.084482      0.9686  0.820790  1.143028                      NA\r
523383  X_155802175_A_G   rs1764581        X  155802175.0               A                G -0.051643       0.949668  ...              T  0.07081  0.051643        0.028584      1.0530  0.995627  1.113679                      NA\r
523384  X_155863130_C_T   rs6567787        X  155863130.0               C                T -0.064325       0.937700  ...              T  0.05985 -0.064325        0.034181      0.9377  0.876937  1.002673                      NA\r
523385  X_155988327_A_C   rs5983854        X  155988327.0               A                C  0.047837       1.049000  ...              C  0.09744  0.047837        0.028863      1.0490  0.991304  1.110054                      NA\r

[518288 rows x 24 columns]
'''

#need to type cast, etc.
remap_dict = {'X':23}
dubois_cd['hm_chrom'] = dubois_cd['hm_chrom'].replace(remap_dict)
dubois_cd['hm_chrom'] = dubois_cd['hm_chrom'].astype('int64')
dubois_cd['hm_pos'] = dubois_cd['hm_pos'].astype('int64')
dubois_cd = dubois_cd[dubois_columns].sort_values(by=['hm_chrom', 'hm_pos'])

'''
dubois_cd
          hm_variant_id     hm_rsid  hm_chrom     hm_pos hm_other_allele hm_effect_allele   hm_beta  hm_odds_ratio  hm_ci_lower  hm_ci_upper  p_value
208495     1_833068_G_A  rs12562034         1     833068               G                A  0.035367       1.036000     0.948992     1.130985  0.42940
186498    1_1070426_C_T   rs3934834         1    1070426               C                T -0.002303       0.997700     0.931117     1.069045  0.94790
208496    1_1083324_A_G   rs9442372         1    1083324               A                G  0.050031       1.051304     0.993990     1.111922  0.08025
186499    1_1086035_A_G   rs3737728         1    1086035               A                G  0.039781       1.040583     0.983784     1.100661  0.16480
186500    1_1095185_C_T   rs6687776         1    1095185               C                T -0.056888       0.944700     0.882960     1.010757  0.09900
...                 ...         ...       ...        ...             ...              ...       ...            ...          ...          ...      ...
523383  X_155802175_A_G   rs1764581        23  155802175               A                G -0.051643       0.949668     0.897925     1.004392  0.07081
523369  X_155863130_C_T   rs6567787        23  155863130               C                T -0.064325       0.937700     0.876937     1.002673  0.05985
523384  X_155863130_C_T   rs6567787        23  155863130               C                T -0.064325       0.937700     0.876937     1.002673  0.05985
523370  X_155988327_A_C   rs5983854        23  155988327               A                C  0.047837       1.049000     0.991304     1.110054  0.09744
523385  X_155988327_A_C   rs5983854        23  155988327               A                C  0.047837       1.049000     0.991304     1.110054  0.09744

[518288 rows x 11 columns]
'''
dubois_cd.equals(dubois_cd.sort_values(by=['hm_chrom', 'hm_pos']))
oc_dubois = ['hm_chrom', 'hm_pos','hm_other_allele', 'hm_effect_allele', 'hm_beta', 'p_value']

'''
dubois_cd[oc_dubois].to_csv('dubois_cd_gwas_lz'+version+'.txt', sep='\t', index=False)
os.system("gzip -k " + 'dubois_cd_gwas_lz'+version+'.txt &')
'''

'''
Study #12: Liu Inflammatory Bowel Disease
http://ftp.ebi.ac.uk/pub/databases/gwas/summary_statistics/GCST003001-GCST004000/GCST003043/harmonised/26192919-GCST003043-EFO_0003767.h.tsv.gz
26192919
'''
LiuIBD = sumstatspath + "26192919-GCST003043-EFO_0003767.h.tsv"
liu_ibd = pd.read_csv(LiuIBD, lineterminator='\n', sep='\t')
liu_ibd.columns.to_list()

liu_ibd.isna().sum()
pd.unique(liu_ibd.hm_code)

'''
liu_ibd.isna().sum()
hm_variant_id                  14465
hm_rsid                        14465
hm_chrom                       14465
hm_pos                         14465
hm_other_allele                14465
hm_effect_allele               14465
hm_beta                        14465
hm_odds_ratio                 126096
hm_ci_lower                   126096
hm_ci_upper                   126096
hm_effect_allele_frequency     14465
hm_code                            0
variant_id                         0
p_value                            0
chromosome                         0
odds_ratio                    126096
base_pair_location                 0
effect_allele                      0
other_allele                       0
effect_allele_frequency            0
beta                               0
range                         126096
standard_error                     0
ci_upper                      126096
ci_lower\r                         0
dtype: int64
>>> pd.unique(liu_ibd.hm_code)
array([11, 12, 10,  9, 13, 14, 15])
'''

liuibd_columns = ['hm_variant_id', 'hm_rsid', 'hm_chrom', 'hm_pos', 'hm_other_allele', 'hm_effect_allele', 'hm_beta', 'hm_odds_ratio', 'hm_ci_lower', 'hm_ci_upper', 'p_value', 'standard_error']

'''
liu_ibd
          hm_variant_id      hm_rsid  hm_chrom       hm_pos hm_other_allele hm_effect_allele   hm_beta  hm_odds_ratio  ...  effect_allele  other_allele  effect_allele_frequency      beta range  standard_error  ci_upper  ci_lower\r
0         10_251693_A_G    rs3125037      10.0     251693.0               A                G -0.003839            NaN  ...              A             G                 0.721900  0.003839   NaN        0.011536       NaN        NA\r
1         10_308361_C_T   rs35198327      10.0     308361.0               C                T -0.003123            NaN  ...              A             G                 0.515200 -0.003123   NaN        0.010457       NaN        NA\r
2         10_318459_A_G    rs9804310      10.0     318459.0               A                G  0.000012            NaN  ...              A             G                 0.858300 -0.000012   NaN        0.014886       NaN        NA\r
3         10_361644_G_A    rs3922851      10.0     361644.0               G                A -0.002016            NaN  ...              A             G                 0.368900 -0.002016   NaN        0.010454       NaN        NA\r
4         10_475246_C_T   rs11252630      10.0     475246.0               C                T -0.007243            NaN  ...              A             G                 0.459300 -0.007243   NaN        0.010147       NaN        NA\r
...                 ...          ...       ...          ...             ...              ...       ...            ...  ...            ...           ...                      ...       ...   ...             ...       ...         ...
126091  9_121338755_G_A   rs79775811       9.0  121338755.0               G                A -0.036474            NaN  ...              A             G                 0.005942 -0.036474   NaN        0.065938       NaN        NA\r
126092  9_136268460_C_T  rs187817977       9.0  136268460.0               C                T  0.024256            NaN  ...              A             G                 0.007963  0.024256   NaN        0.068256       NaN        NA\r
126093              NaN          NaN       NaN          NaN             NaN              NaN       NaN            NaN  ...              A             T                 0.004799  0.063674   NaN        0.082037       NaN        NA\r
126094  9_136342119_G_A  rs116909701       9.0  136342119.0               G                A  0.184648            NaN  ...              A             G                 0.002972  0.184648   NaN        0.104221       NaN        NA\r
126095  9_136395979_G_A   rs12344738       9.0  136395979.0               G                A  0.040403            NaN  ...              A             G                 0.000654  0.040403   NaN        0.047508       NaN        NA\r

[126096 rows x 25 columns]
'''

#Drop hm_variant_id NA
#no hm_odds_ratio that is not NaN
#DROP hm_beta NA
#do not drop hm_odds_ratio due to ... all of them are NaN
liu_ibd.dropna(axis=0, subset=['hm_variant_id', 'hm_beta'], inplace=True)

'''
liu_ibd
          hm_variant_id      hm_rsid  hm_chrom       hm_pos hm_other_allele hm_effect_allele   hm_beta  hm_odds_ratio  ...  effect_allele  other_allele  effect_allele_frequency      beta range  standard_error  ci_upper  ci_lower\r
0         10_251693_A_G    rs3125037      10.0     251693.0               A                G -0.003839            NaN  ...              A             G                 0.721900  0.003839   NaN        0.011536       NaN        NA\r
1         10_308361_C_T   rs35198327      10.0     308361.0               C                T -0.003123            NaN  ...              A             G                 0.515200 -0.003123   NaN        0.010457       NaN        NA\r
2         10_318459_A_G    rs9804310      10.0     318459.0               A                G  0.000012            NaN  ...              A             G                 0.858300 -0.000012   NaN        0.014886       NaN        NA\r
3         10_361644_G_A    rs3922851      10.0     361644.0               G                A -0.002016            NaN  ...              A             G                 0.368900 -0.002016   NaN        0.010454       NaN        NA\r
4         10_475246_C_T   rs11252630      10.0     475246.0               C                T -0.007243            NaN  ...              A             G                 0.459300 -0.007243   NaN        0.010147       NaN        NA\r
...                 ...          ...       ...          ...             ...              ...       ...            ...  ...            ...           ...                      ...       ...   ...             ...       ...         ...
126090  9_121286194_C_T  rs111298625       9.0  121286194.0               C                T  0.017032            NaN  ...              A             G                 0.028150  0.017032   NaN        0.030780       NaN        NA\r
126091  9_121338755_G_A   rs79775811       9.0  121338755.0               G                A -0.036474            NaN  ...              A             G                 0.005942 -0.036474   NaN        0.065938       NaN        NA\r
126092  9_136268460_C_T  rs187817977       9.0  136268460.0               C                T  0.024256            NaN  ...              A             G                 0.007963  0.024256   NaN        0.068256       NaN        NA\r
126094  9_136342119_G_A  rs116909701       9.0  136342119.0               G                A  0.184648            NaN  ...              A             G                 0.002972  0.184648   NaN        0.104221       NaN        NA\r
126095  9_136395979_G_A   rs12344738       9.0  136395979.0               G                A  0.040403            NaN  ...              A             G                 0.000654  0.040403   NaN        0.047508       NaN        NA\r

[111631 rows x 25 columns]
'''

#Drop invalid hm_codes
print(len(liu_ibd.index))
111631
hm_code_mask = liu_ibd.hm_code.ge(14) | liu_ibd.hm_code.le(6)
hm_code_mask.sum()
0
liu_ibd.drop(liu_ibd[hm_code_mask].index, inplace=True)
print(len(liu_ibd.index))
111631

'''
liu_ibd
          hm_variant_id      hm_rsid  hm_chrom       hm_pos hm_other_allele hm_effect_allele   hm_beta  hm_odds_ratio  ...  effect_allele  other_allele  effect_allele_frequency      beta range  standard_error  ci_upper  ci_lower\r
0         10_251693_A_G    rs3125037      10.0     251693.0               A                G -0.003839            NaN  ...              A             G                 0.721900  0.003839   NaN        0.011536       NaN        NA\r
1         10_308361_C_T   rs35198327      10.0     308361.0               C                T -0.003123            NaN  ...              A             G                 0.515200 -0.003123   NaN        0.010457       NaN        NA\r
2         10_318459_A_G    rs9804310      10.0     318459.0               A                G  0.000012            NaN  ...              A             G                 0.858300 -0.000012   NaN        0.014886       NaN        NA\r
3         10_361644_G_A    rs3922851      10.0     361644.0               G                A -0.002016            NaN  ...              A             G                 0.368900 -0.002016   NaN        0.010454       NaN        NA\r
4         10_475246_C_T   rs11252630      10.0     475246.0               C                T -0.007243            NaN  ...              A             G                 0.459300 -0.007243   NaN        0.010147       NaN        NA\r
...                 ...          ...       ...          ...             ...              ...       ...            ...  ...            ...           ...                      ...       ...   ...             ...       ...         ...
126090  9_121286194_C_T  rs111298625       9.0  121286194.0               C                T  0.017032            NaN  ...              A             G                 0.028150  0.017032   NaN        0.030780       NaN        NA\r
126091  9_121338755_G_A   rs79775811       9.0  121338755.0               G                A -0.036474            NaN  ...              A             G                 0.005942 -0.036474   NaN        0.065938       NaN        NA\r
126092  9_136268460_C_T  rs187817977       9.0  136268460.0               C                T  0.024256            NaN  ...              A             G                 0.007963  0.024256   NaN        0.068256       NaN        NA\r
126094  9_136342119_G_A  rs116909701       9.0  136342119.0               G                A  0.184648            NaN  ...              A             G                 0.002972  0.184648   NaN        0.104221       NaN        NA\r
126095  9_136395979_G_A   rs12344738       9.0  136395979.0               G                A  0.040403            NaN  ...              A             G                 0.000654  0.040403   NaN        0.047508       NaN        NA\r

[111631 rows x 25 columns]
'''
#need to type cast, etc.
remap_dict = {'X':23}
liu_ibd['hm_chrom'] = liu_ibd['hm_chrom'].replace(remap_dict)
liu_ibd['hm_chrom'] = liu_ibd['hm_chrom'].astype('int64')
liu_ibd['hm_pos'] = liu_ibd['hm_pos'].astype('int64')
liu_ibd = liu_ibd[liuibd_columns].sort_values(by=['hm_chrom', 'hm_pos'])

'''
liu_ibd
         hm_variant_id     hm_rsid  hm_chrom    hm_pos hm_other_allele hm_effect_allele   hm_beta  hm_odds_ratio  hm_ci_lower  hm_ci_upper       p_value
40782    1_1182895_C_T  rs61733845         1   1182895               C                T -0.005002            NaN          NaN          NaN  8.447717e-01
40783    1_1185051_G_A   rs1320571         1   1185051               G                A -0.007279            NaN          NaN          NaN  7.719240e-01
40784    1_1199862_A_C   rs9729550         1   1199862               A                C  0.047465            NaN          NaN          NaN  5.353847e-05
40785    1_1205055_G_T   rs1815606         1   1205055               G                T  0.041764            NaN          NaN          NaN  2.475336e-04
40786    1_1228424_C_T   rs7515488         1   1228424               C                T  0.087431            NaN          NaN          NaN  2.852893e-10
...                ...         ...       ...       ...             ...              ...       ...            ...          ...          ...           ...
59928  22_50656498_C_T   rs9616810        22  50656498               C                T  0.016225            NaN          NaN          NaN  1.816897e-01
59929  22_50667128_C_T   rs9616812        22  50667128               C                T -0.007782            NaN          NaN          NaN  4.493614e-01
59930  22_50671564_T_C   rs9628185        22  50671564               T                C -0.007534            NaN          NaN          NaN  4.683724e-01
60230  22_50695758_A_G   rs8135777        22  50695758               A                G -0.316557            NaN          NaN          NaN  2.132347e-01
59931  22_50718238_C_T   rs9628187        22  50718238               C                T -0.003751            NaN          NaN          NaN  7.741766e-01

[111631 rows x 11 columns]
'''
liu_ibd.equals(liu_ibd.sort_values(by=['hm_chrom', 'hm_pos']))
oc_liu = ['hm_chrom', 'hm_pos','hm_other_allele', 'hm_effect_allele', 'hm_beta', 'p_value']

'''
liu_ibd[oc_liu].to_csv('liu_ibd_gwas_lz'+version+'.txt', sep='\t', index=False)
os.system("gzip -k " + 'liu_ibd_gwas_lz'+version+'.txt &')
'''

'''
Study #13: Liu Crohn's Disease
http://ftp.ebi.ac.uk/pub/databases/gwas/summary_statistics/GCST003001-GCST004000/GCST003044/harmonised/26192919-GCST003044-EFO_0000384.h.tsv.gz
26192919
'''
LiuCD = sumstatspath + "26192919-GCST003044-EFO_0000384.h.tsv"
liu_cd = pd.read_csv(LiuCD, lineterminator='\n', sep='\t')
liu_cd.columns.to_list()

liu_cd.isna().sum()
pd.unique(liu_cd.hm_code)

'''
liu_cd.isna().sum()
hm_variant_id                  14302
hm_rsid                        14302
hm_chrom                       14302
hm_pos                         14302
hm_other_allele                14302
hm_effect_allele               14302
hm_beta                        14302
hm_odds_ratio                 124885
hm_ci_lower                   124885
hm_ci_upper                   124885
hm_effect_allele_frequency     14302
hm_code                            0
variant_id                         0
p_value                            0
chromosome                         0
odds_ratio                    124885
base_pair_location                 0
effect_allele                      0
other_allele                       0
effect_allele_frequency            0
beta                               0
range                         124885
standard_error                     0
ci_lower                      124885
ci_upper\r                         0
dtype: int64
>>> pd.unique(liu_cd.hm_code)
array([11, 12, 10,  9, 13, 15, 14])
'''

liucd_columns = ['hm_variant_id', 'hm_rsid', 'hm_chrom', 'hm_pos', 'hm_other_allele', 'hm_effect_allele', 'hm_beta', 'hm_odds_ratio', 'hm_ci_lower', 'hm_ci_upper', 'p_value', 'standard_error']

'''
liu_cd
          hm_variant_id      hm_rsid  hm_chrom       hm_pos hm_other_allele hm_effect_allele   hm_beta  hm_odds_ratio  ...  effect_allele  other_allele  effect_allele_frequency      beta range  standard_error  ci_lower  ci_upper\r
0         10_251693_A_G    rs3125037      10.0     251693.0               A                G -0.009213            NaN  ...              A             G                 0.721900  0.009213   NaN        0.013971       NaN        NA\r
1         10_308361_C_T   rs35198327      10.0     308361.0               C                T -0.010199            NaN  ...              A             G                 0.515200 -0.010199   NaN        0.012674       NaN        NA\r
2         10_318459_A_G    rs9804310      10.0     318459.0               A                G -0.001511            NaN  ...              A             G                 0.858300  0.001511   NaN        0.017976       NaN        NA\r
3         10_361644_G_A    rs3922851      10.0     361644.0               G                A  0.000475            NaN  ...              A             G                 0.368900  0.000475   NaN        0.012673       NaN        NA\r
4         10_475246_C_T   rs11252630      10.0     475246.0               C                T  0.002036            NaN  ...              A             G                 0.459300  0.002036   NaN        0.012288       NaN        NA\r
...                 ...          ...       ...          ...             ...              ...       ...            ...  ...            ...           ...                      ...       ...   ...             ...       ...         ...
124880  9_136416026_C_T  rs117166540       9.0  136416026.0               C                T -0.153877            NaN  ...              A             G                 0.014530 -0.153877   NaN        0.055090       NaN        NA\r
124881  9_136427000_C_T    rs4436242       9.0  136427000.0               C                T  0.181913            NaN  ...              A             G                 0.000639  0.181913   NaN        0.202920       NaN        NA\r
124882  9_136437886_T_C   rs78558933       9.0  136437886.0               T                C  0.192118            NaN  ...              A             G                 0.989750 -0.192118   NaN        0.056290       NaN        NA\r
124883              NaN          NaN       NaN          NaN             NaN              NaN       NaN            NaN  ...              C             G                 0.009660  0.180038   NaN        0.058816       NaN        NA\r
124884  9_136470321_C_T  rs115189185       9.0  136470321.0               C                T  0.168861            NaN  ...              A             G                 0.009918  0.168861   NaN        0.057496       NaN        NA\r

[124885 rows x 25 columns]
'''

#Drop hm_variant_id NA
#no hm_odds_ratio that is not NaN
#DROP hm_beta NA
#do not drop hm_odds_ratio due to ... all of them are NaN
liu_cd.dropna(axis=0, subset=['hm_variant_id', 'hm_beta'], inplace=True)

'''
liu_cd
          hm_variant_id      hm_rsid  hm_chrom       hm_pos hm_other_allele hm_effect_allele   hm_beta  hm_odds_ratio  ...  effect_allele  other_allele  effect_allele_frequency      beta range  standard_error  ci_lower  ci_upper\r
0         10_251693_A_G    rs3125037      10.0     251693.0               A                G -0.009213            NaN  ...              A             G                 0.721900  0.009213   NaN        0.013971       NaN        NA\r
1         10_308361_C_T   rs35198327      10.0     308361.0               C                T -0.010199            NaN  ...              A             G                 0.515200 -0.010199   NaN        0.012674       NaN        NA\r
2         10_318459_A_G    rs9804310      10.0     318459.0               A                G -0.001511            NaN  ...              A             G                 0.858300  0.001511   NaN        0.017976       NaN        NA\r
3         10_361644_G_A    rs3922851      10.0     361644.0               G                A  0.000475            NaN  ...              A             G                 0.368900  0.000475   NaN        0.012673       NaN        NA\r
4         10_475246_C_T   rs11252630      10.0     475246.0               C                T  0.002036            NaN  ...              A             G                 0.459300  0.002036   NaN        0.012288       NaN        NA\r
...                 ...          ...       ...          ...             ...              ...       ...            ...  ...            ...           ...                      ...       ...   ...             ...       ...         ...
124879  9_121353315_T_C  rs116871862       9.0  121353315.0               T                C  0.095665            NaN  ...              A             G                 0.985810 -0.095665   NaN        0.054305       NaN        NA\r
124880  9_136416026_C_T  rs117166540       9.0  136416026.0               C                T -0.153877            NaN  ...              A             G                 0.014530 -0.153877   NaN        0.055090       NaN        NA\r
124881  9_136427000_C_T    rs4436242       9.0  136427000.0               C                T  0.181913            NaN  ...              A             G                 0.000639  0.181913   NaN        0.202920       NaN        NA\r
124882  9_136437886_T_C   rs78558933       9.0  136437886.0               T                C  0.192118            NaN  ...              A             G                 0.989750 -0.192118   NaN        0.056290       NaN        NA\r
124884  9_136470321_C_T  rs115189185       9.0  136470321.0               C                T  0.168861            NaN  ...              A             G                 0.009918  0.168861   NaN        0.057496       NaN        NA\r

[110583 rows x 25 columns]
'''

#Drop invalid hm_codes
print(len(liu_cd.index))
110583
hm_code_mask = liu_cd.hm_code.ge(14) | liu_cd.hm_code.le(6)
hm_code_mask.sum()
0
liu_ibd.drop(liu_cd[hm_code_mask].index, inplace=True)
print(len(liu_cd.index))
110583

'''
liu_cd
          hm_variant_id      hm_rsid  hm_chrom       hm_pos hm_other_allele hm_effect_allele   hm_beta  hm_odds_ratio  ...  effect_allele  other_allele  effect_allele_frequency      beta range  standard_error  ci_lower  ci_upper\r
0         10_251693_A_G    rs3125037      10.0     251693.0               A                G -0.009213            NaN  ...              A             G                 0.721900  0.009213   NaN        0.013971       NaN        NA\r
1         10_308361_C_T   rs35198327      10.0     308361.0               C                T -0.010199            NaN  ...              A             G                 0.515200 -0.010199   NaN        0.012674       NaN        NA\r
2         10_318459_A_G    rs9804310      10.0     318459.0               A                G -0.001511            NaN  ...              A             G                 0.858300  0.001511   NaN        0.017976       NaN        NA\r
3         10_361644_G_A    rs3922851      10.0     361644.0               G                A  0.000475            NaN  ...              A             G                 0.368900  0.000475   NaN        0.012673       NaN        NA\r
4         10_475246_C_T   rs11252630      10.0     475246.0               C                T  0.002036            NaN  ...              A             G                 0.459300  0.002036   NaN        0.012288       NaN        NA\r
...                 ...          ...       ...          ...             ...              ...       ...            ...  ...            ...           ...                      ...       ...   ...             ...       ...         ...
124879  9_121353315_T_C  rs116871862       9.0  121353315.0               T                C  0.095665            NaN  ...              A             G                 0.985810 -0.095665   NaN        0.054305       NaN        NA\r
124880  9_136416026_C_T  rs117166540       9.0  136416026.0               C                T -0.153877            NaN  ...              A             G                 0.014530 -0.153877   NaN        0.055090       NaN        NA\r
124881  9_136427000_C_T    rs4436242       9.0  136427000.0               C                T  0.181913            NaN  ...              A             G                 0.000639  0.181913   NaN        0.202920       NaN        NA\r
124882  9_136437886_T_C   rs78558933       9.0  136437886.0               T                C  0.192118            NaN  ...              A             G                 0.989750 -0.192118   NaN        0.056290       NaN        NA\r
124884  9_136470321_C_T  rs115189185       9.0  136470321.0               C                T  0.168861            NaN  ...              A             G                 0.009918  0.168861   NaN        0.057496       NaN        NA\r

[110583 rows x 25 columns]
'''
#need to type cast, etc.
remap_dict = {'X':23}
liu_cd['hm_chrom'] = liu_cd['hm_chrom'].replace(remap_dict)
liu_cd['hm_chrom'] = liu_cd['hm_chrom'].astype('int64')
liu_cd['hm_pos'] = liu_cd['hm_pos'].astype('int64')
liu_cd = liu_cd[liucd_columns].sort_values(by=['hm_chrom', 'hm_pos'])

'''
liu_cd
         hm_variant_id     hm_rsid  hm_chrom    hm_pos hm_other_allele hm_effect_allele   hm_beta  hm_odds_ratio  hm_ci_lower  hm_ci_upper       p_value
40342    1_1182895_C_T  rs61733845         1   1182895               C                T -0.029500            NaN          NaN          NaN  3.374010e-01
40343    1_1185051_G_A   rs1320571         1   1185051               G                A -0.022995            NaN          NaN          NaN  4.450345e-01
40344    1_1199862_A_C   rs9729550         1   1199862               A                C  0.052494            NaN          NaN          NaN  1.979683e-04
40345    1_1205055_G_T   rs1815606         1   1205055               G                T  0.043250            NaN          NaN          NaN  1.545641e-03
40346    1_1228424_C_T   rs7515488         1   1228424               C                T  0.084067            NaN          NaN          NaN  5.426972e-07
...                ...         ...       ...       ...             ...              ...       ...            ...          ...          ...           ...
59301  22_50639823_C_T   rs4040041        22  50639823               C                T -0.007504            NaN          NaN          NaN  5.502236e-01
59302  22_50656498_C_T   rs9616810        22  50656498               C                T -0.003659            NaN          NaN          NaN  8.043222e-01
59303  22_50667128_C_T   rs9616812        22  50667128               C                T  0.005323            NaN          NaN          NaN  6.692001e-01
59304  22_50671564_T_C   rs9628185        22  50671564               T                C  0.009027            NaN          NaN          NaN  4.700148e-01
59305  22_50718238_C_T   rs9628187        22  50718238               C                T -0.001878            NaN          NaN          NaN  9.052720e-01

[110583 rows x 11 columns]
'''

liu_cd.equals(liu_cd.sort_values(by=['hm_chrom', 'hm_pos']))
oc_liu = ['hm_chrom', 'hm_pos','hm_other_allele', 'hm_effect_allele', 'hm_beta', 'p_value']

'''
liu_cd[oc_liu].to_csv('liu_cd_gwas_lz'+version+'.txt', sep='\t', index=False)
os.system("gzip -k " + 'liu_cd_gwas_lz'+version+'.txt &')
'''

'''
Study #14: deLange Inflammatory Bowel Disease
http://ftp.ebi.ac.uk/pub/databases/gwas/summary_statistics/GCST004001-GCST005000/GCST004131/harmonised/28067908-GCST004131-EFO_0003767.h.tsv.gz
28067908
'''
deLangeIBD = sumstatspath + "28067908-GCST004131-EFO_0003767.h.tsv"
delange_ibd = pd.read_csv(deLangeIBD, lineterminator='\n', sep='\t')
delange_ibd.columns.to_list()

delange_ibd.isna().sum()
pd.unique(delange_ibd.hm_code)

'''
delange_ibd.isna().sum()
hm_variant_id                       0
hm_rsid                             0
hm_chrom                            0
hm_pos                              0
hm_other_allele                     0
hm_effect_allele                    0
hm_beta                         14614
hm_odds_ratio                 9633630
hm_ci_lower                   9633630
hm_ci_upper                   9633630
hm_effect_allele_frequency    9633630
hm_code                             0
other_allele                        0
effect_allele                       0
beta                                0
standard_error                      0
p_value                             0
chromosome                          0
base_pair_location                  0
odds_ratio                    9633630
ci_lower                      9633630
variant_id                          0
effect_allele_frequency       9633630
ci_upper\r                          0
dtype: int64
>>> pd.unique(delange_ibd.hm_code)
array([10, 11,  6,  5, 12, 13])
'''

delangeibd_columns = ['hm_variant_id', 'hm_rsid', 'hm_chrom', 'hm_pos', 'hm_other_allele', 'hm_effect_allele', 'hm_beta', 'hm_odds_ratio', 'hm_ci_lower', 'hm_ci_upper', 'p_value', 'standard_error']

'''
delange_ibd
           hm_variant_id      hm_rsid  hm_chrom    hm_pos hm_other_allele hm_effect_allele  hm_beta  hm_odds_ratio  ...  p_value  chromosome  base_pair_location  odds_ratio ci_lower   variant_id  effect_allele_frequency  ci_upper\r
0         10_9958055_A_G    rs6602381        10   9958055               A                G   0.0108            NaN  ...  0.39360          10             9958055         NaN      NaN    rs6602381                      NaN        NA\r
1        10_98240868_A_G    rs7899632        10  98240868               A                G   0.0230            NaN  ...  0.06407          10            98240868         NaN      NaN    rs7899632                      NaN        NA\r
2        10_98240888_A_C   rs61875309        10  98240888               A                C  -0.0291            NaN  ...  0.05748          10            98240888         NaN      NaN   rs61875309                      NaN        NA\r
3        10_98242110_C_T  rs150203744        10  98242110               C                T   0.0607            NaN  ...  0.29600          10            98242110         NaN      NaN  rs150203744                      NaN        NA\r
4        10_98242707_T_C  rs111551711        10  98242707               T                C  -0.0225            NaN  ...  0.75790          10            98242707         NaN      NaN  rs111551711                      NaN        NA\r
...                  ...          ...       ...       ...             ...              ...      ...            ...  ...      ...         ...                 ...         ...      ...          ...                      ...         ...
9633625   9_97236364_C_G   rs10981297         9  97236364               C                G  -0.0002            NaN  ...  0.98750           9            97236364         NaN      NaN   rs10981297                      NaN        NA\r
9633626    9_9999880_G_C  rs151001359         9   9999880               G                C   0.1749            NaN  ...  0.03586           9             9999880         NaN      NaN  rs151001359                      NaN        NA\r
9633627   9_97236872_G_A   rs80110029         9  97236872               G                A  -0.0873            NaN  ...  0.50520           9            97236872         NaN      NaN   rs80110029                      NaN        NA\r
9633628   9_97237186_A_G   rs10981301         9  97237186               A                G  -0.0069            NaN  ...  0.74750           9            97237186         NaN      NaN   rs10981301                      NaN        NA\r
9633629  9_97237543_A_AC  rs148363074         9  97237543               A               AC  -0.1876            NaN  ...  0.22310           9            97237543         NaN      NaN  rs148363074                      NaN        NA\r

[9633630 rows x 24 columns]
'''

#Drop hm_variant_id NA
#no hm_odds_ratio that is not NaN
#DROP hm_beta NA
#do not drop hm_odds_ratio due to ... all of them are NaN
delange_ibd.dropna(axis=0, subset=['hm_variant_id', 'hm_beta'], inplace=True)

'''
delange_ibd
           hm_variant_id      hm_rsid  hm_chrom    hm_pos hm_other_allele hm_effect_allele  hm_beta  hm_odds_ratio  ...  p_value  chromosome  base_pair_location  odds_ratio ci_lower   variant_id  effect_allele_frequency  ci_upper\r
0         10_9958055_A_G    rs6602381        10   9958055               A                G   0.0108            NaN  ...  0.39360          10             9958055         NaN      NaN    rs6602381                      NaN        NA\r
1        10_98240868_A_G    rs7899632        10  98240868               A                G   0.0230            NaN  ...  0.06407          10            98240868         NaN      NaN    rs7899632                      NaN        NA\r
2        10_98240888_A_C   rs61875309        10  98240888               A                C  -0.0291            NaN  ...  0.05748          10            98240888         NaN      NaN   rs61875309                      NaN        NA\r
3        10_98242110_C_T  rs150203744        10  98242110               C                T   0.0607            NaN  ...  0.29600          10            98242110         NaN      NaN  rs150203744                      NaN        NA\r
4        10_98242707_T_C  rs111551711        10  98242707               T                C  -0.0225            NaN  ...  0.75790          10            98242707         NaN      NaN  rs111551711                      NaN        NA\r
...                  ...          ...       ...       ...             ...              ...      ...            ...  ...      ...         ...                 ...         ...      ...          ...                      ...         ...
9633625   9_97236364_C_G   rs10981297         9  97236364               C                G  -0.0002            NaN  ...  0.98750           9            97236364         NaN      NaN   rs10981297                      NaN        NA\r
9633626    9_9999880_G_C  rs151001359         9   9999880               G                C   0.1749            NaN  ...  0.03586           9             9999880         NaN      NaN  rs151001359                      NaN        NA\r
9633627   9_97236872_G_A   rs80110029         9  97236872               G                A  -0.0873            NaN  ...  0.50520           9            97236872         NaN      NaN   rs80110029                      NaN        NA\r
9633628   9_97237186_A_G   rs10981301         9  97237186               A                G  -0.0069            NaN  ...  0.74750           9            97237186         NaN      NaN   rs10981301                      NaN        NA\r
9633629  9_97237543_A_AC  rs148363074         9  97237543               A               AC  -0.1876            NaN  ...  0.22310           9            97237543         NaN      NaN  rs148363074                      NaN        NA\r

[9619016 rows x 24 columns]
'''

#Drop invalid hm_codes
print(len(delange_ibd.index))
9619016
hm_code_mask = delange_ibd.hm_code.ge(14) | delange_ibd.hm_code.le(6)
hm_code_mask.sum()
1385931
delange_ibd.drop(delange_ibd[hm_code_mask].index, inplace=True)
print(len(delange_ibd.index))
8233085

'''
delange_ibd
           hm_variant_id      hm_rsid  hm_chrom    hm_pos hm_other_allele hm_effect_allele  hm_beta  hm_odds_ratio  ...  p_value  chromosome  base_pair_location  odds_ratio ci_lower   variant_id  effect_allele_frequency  ci_upper\r
0         10_9958055_A_G    rs6602381        10   9958055               A                G   0.0108            NaN  ...  0.39360          10             9958055         NaN      NaN    rs6602381                      NaN        NA\r
1        10_98240868_A_G    rs7899632        10  98240868               A                G   0.0230            NaN  ...  0.06407          10            98240868         NaN      NaN    rs7899632                      NaN        NA\r
2        10_98240888_A_C   rs61875309        10  98240888               A                C  -0.0291            NaN  ...  0.05748          10            98240888         NaN      NaN   rs61875309                      NaN        NA\r
3        10_98242110_C_T  rs150203744        10  98242110               C                T   0.0607            NaN  ...  0.29600          10            98242110         NaN      NaN  rs150203744                      NaN        NA\r
4        10_98242707_T_C  rs111551711        10  98242707               T                C  -0.0225            NaN  ...  0.75790          10            98242707         NaN      NaN  rs111551711                      NaN        NA\r
...                  ...          ...       ...       ...             ...              ...      ...            ...  ...      ...         ...                 ...         ...      ...          ...                      ...         ...
9633623   9_97235425_C_T   rs11794422         9  97235425               C                T  -0.0010            NaN  ...  0.93630           9            97235425         NaN      NaN   rs11794422                      NaN        NA\r
9633624   9_97236121_C_T   rs10981296         9  97236121               C                T  -0.0021            NaN  ...  0.86430           9            97236121         NaN      NaN   rs10981296                      NaN        NA\r
9633627   9_97236872_G_A   rs80110029         9  97236872               G                A  -0.0873            NaN  ...  0.50520           9            97236872         NaN      NaN   rs80110029                      NaN        NA\r
9633628   9_97237186_A_G   rs10981301         9  97237186               A                G  -0.0069            NaN  ...  0.74750           9            97237186         NaN      NaN   rs10981301                      NaN        NA\r
9633629  9_97237543_A_AC  rs148363074         9  97237543               A               AC  -0.1876            NaN  ...  0.22310           9            97237543         NaN      NaN  rs148363074                      NaN        NA\r

[8233085 rows x 24 columns]
'''
#need to type cast, etc.
remap_dict = {'X':23}
delange_ibd['hm_chrom'] = delange_ibd['hm_chrom'].replace(remap_dict)
delange_ibd['hm_chrom'] = delange_ibd['hm_chrom'].astype('int64')
delange_ibd['hm_pos'] = delange_ibd['hm_pos'].astype('int64')
delange_ibd = delange_ibd[delangeibd_columns].sort_values(by=['hm_chrom', 'hm_pos'])

'''
delange_ibd
           hm_variant_id      hm_rsid  hm_chrom    hm_pos hm_other_allele hm_effect_allele  hm_beta  hm_odds_ratio  hm_ci_lower  hm_ci_upper   p_value
4037549     1_727242_G_A   rs61769339         1    727242               G                A  -0.0258            NaN          NaN          NaN  0.305200
4049089     1_758351_A_G   rs12238997         1    758351               A                G  -0.0218            NaN          NaN          NaN  0.367300
4051832     1_766455_T_C  rs189800799         1    766455               T                C  -0.0859            NaN          NaN          NaN  0.065680
4055493     1_777550_T_C   rs28457007         1    777550               T                C  -0.0744            NaN          NaN          NaN  0.144700
4055913     1_778639_A_G  rs114983708         1    778639               A                G  -0.0505            NaN          NaN          NaN  0.214100
...                  ...          ...       ...       ...             ...              ...      ...            ...          ...          ...       ...
4639445  22_50791377_T_C    rs9616985        22  50791377               T                C  -0.0112            NaN          NaN          NaN  0.679800
4639446  22_50791427_G_A  rs144549712        22  50791427               G                A   0.0168            NaN          NaN          NaN  0.457200
4639447  22_50796371_G_A  rs191117135        22  50796371               G                A   0.1933            NaN          NaN          NaN  0.005552
4639448  22_50798635_T_C    rs3896457        22  50798635               T                C   0.0005            NaN          NaN          NaN  0.972300
4639449  22_50799821_A_C  rs149733995        22  50799821               A                C   0.0016            NaN          NaN          NaN  0.953800

[8233085 rows x 11 columns]
'''
delange_ibd.equals(delange_ibd.sort_values(by=['hm_chrom', 'hm_pos']))
oc_delange = ['hm_chrom', 'hm_pos','hm_other_allele', 'hm_effect_allele', 'hm_beta', 'p_value']

'''
delange_ibd[oc_delange].to_csv('delange_ibd_gwas_lz'+version+'.txt', sep='\t', index=False)
os.system("gzip -k " + 'delange_ibd_gwas_lz'+version+'.txt &')
'''

'''
Study #15: deLange Crohn's Disease
http://ftp.ebi.ac.uk/pub/databases/gwas/summary_statistics/GCST004001-GCST005000/GCST004132/harmonised/28067908-GCST004132-EFO_0000384.h.tsv.gz
28067908
'''
deLangeCD = sumstatspath + "28067908-GCST004132-EFO_0000384.h.tsv"
delange_cd = pd.read_csv(deLangeCD, lineterminator='\n', sep='\t')
delange_cd.columns.to_list()

delange_cd.isna().sum()
pd.unique(delange_cd.hm_code)

'''
delange_cd.isna().sum()
hm_variant_id                       0
hm_rsid                             0
hm_chrom                            0
hm_pos                              0
hm_other_allele                     0
hm_effect_allele                    0
hm_beta                         11594
hm_odds_ratio                 9469592
hm_ci_lower                   9469592
hm_ci_upper                   9469592
hm_effect_allele_frequency    9469592
hm_code                             0
other_allele                        0
effect_allele                       0
beta                                0
standard_error                      0
p_value                             0
chromosome                          0
base_pair_location                  0
odds_ratio                    9469592
ci_lower                      9469592
ci_upper                      9469592
effect_allele_frequency       9469592
variant_id\r                        0
dtype: int64
>>> pd.unique(delange_cd.hm_code)
array([10, 11,  6,  5, 12, 13])
'''

delangecd_columns = ['hm_variant_id', 'hm_rsid', 'hm_chrom', 'hm_pos', 'hm_other_allele', 'hm_effect_allele', 'hm_beta', 'hm_odds_ratio', 'hm_ci_lower', 'hm_ci_upper', 'p_value', 'standard_error']

'''
delange_cd
           hm_variant_id      hm_rsid  hm_chrom    hm_pos hm_other_allele hm_effect_allele  hm_beta  hm_odds_ratio  ...  p_value  chromosome  base_pair_location  odds_ratio ci_lower ci_upper  effect_allele_frequency   variant_id\r
0         10_9958055_A_G    rs6602381        10   9958055               A                G   0.0109            NaN  ...  0.50440          10             9958055         NaN      NaN      NaN                      NaN    rs6602381\r
1        10_98240868_A_G    rs7899632        10  98240868               A                G   0.0181            NaN  ...  0.26050          10            98240868         NaN      NaN      NaN                      NaN    rs7899632\r
2        10_98240888_A_C   rs61875309        10  98240888               A                C  -0.0264            NaN  ...  0.18320          10            98240888         NaN      NaN      NaN                      NaN   rs61875309\r
3        10_98242110_C_T  rs150203744        10  98242110               C                T   0.0877            NaN  ...  0.22670          10            98242110         NaN      NaN      NaN                      NaN  rs150203744\r
4        10_98242707_T_C  rs111551711        10  98242707               T                C  -0.1536            NaN  ...  0.09955          10            98242707         NaN      NaN      NaN                      NaN  rs111551711\r
...                  ...          ...       ...       ...             ...              ...      ...            ...  ...      ...         ...                 ...         ...      ...      ...                      ...            ...
9469587   9_97235425_C_T   rs11794422         9  97235425               C                T   0.0083            NaN  ...  0.60850           9            97235425         NaN      NaN      NaN                      NaN   rs11794422\r
9469588   9_97236121_C_T   rs10981296         9  97236121               C                T   0.0071            NaN  ...  0.65960           9            97236121         NaN      NaN      NaN                      NaN   rs10981296\r
9469589   9_97236364_C_G   rs10981297         9  97236364               C                G   0.0097            NaN  ...  0.63780           9            97236364         NaN      NaN      NaN                      NaN   rs10981297\r
9469590   9_97236872_G_A   rs80110029         9  97236872               G                A  -0.2163            NaN  ...  0.18640           9            97236872         NaN      NaN      NaN                      NaN   rs80110029\r
9469591   9_97237186_A_G   rs10981301         9  97237186               A                G  -0.0171            NaN  ...  0.54390           9            97237186         NaN      NaN      NaN                      NaN   rs10981301\r

[9469592 rows x 24 columns]
'''

#Drop hm_variant_id NA
#no hm_odds_ratio that is not NaN
#DROP hm_beta NA
#do not drop hm_odds_ratio due to ... all of them are NaN
delange_cd.dropna(axis=0, subset=['hm_variant_id', 'hm_beta'], inplace=True)

'''
delange_cd
           hm_variant_id      hm_rsid  hm_chrom    hm_pos hm_other_allele hm_effect_allele  hm_beta  hm_odds_ratio  ...  p_value  chromosome  base_pair_location  odds_ratio ci_lower   variant_id  effect_allele_frequency  ci_upper\r
0         10_9958055_A_G    rs6602381        10   9958055               A                G   0.0108            NaN  ...  0.39360          10             9958055         NaN      NaN    rs6602381                      NaN        NA\r
1        10_98240868_A_G    rs7899632        10  98240868               A                G   0.0230            NaN  ...  0.06407          10            98240868         NaN      NaN    rs7899632                      NaN        NA\r
2        10_98240888_A_C   rs61875309        10  98240888               A                C  -0.0291            NaN  ...  0.05748          10            98240888         NaN      NaN   rs61875309                      NaN        NA\r
3        10_98242110_C_T  rs150203744        10  98242110               C                T   0.0607            NaN  ...  0.29600          10            98242110         NaN      NaN  rs150203744                      NaN        NA\r
4        10_98242707_T_C  rs111551711        10  98242707               T                C  -0.0225            NaN  ...  0.75790          10            98242707         NaN      NaN  rs111551711                      NaN        NA\r
...                  ...          ...       ...       ...             ...              ...      ...            ...  ...      ...         ...                 ...         ...      ...          ...                      ...         ...
9633625   9_97236364_C_G   rs10981297         9  97236364               C                G  -0.0002            NaN  ...  0.98750           9            97236364         NaN      NaN   rs10981297                      NaN        NA\r
9633626    9_9999880_G_C  rs151001359         9   9999880               G                C   0.1749            NaN  ...  0.03586           9             9999880         NaN      NaN  rs151001359                      NaN        NA\r
9633627   9_97236872_G_A   rs80110029         9  97236872               G                A  -0.0873            NaN  ...  0.50520           9            97236872         NaN      NaN   rs80110029                      NaN        NA\r
9633628   9_97237186_A_G   rs10981301         9  97237186               A                G  -0.0069            NaN  ...  0.74750           9            97237186         NaN      NaN   rs10981301                      NaN        NA\r
9633629  9_97237543_A_AC  rs148363074         9  97237543               A               AC  -0.1876            NaN  ...  0.22310           9            97237543         NaN      NaN  rs148363074                      NaN        NA\r

[9619016 rows x 24 columns]
'''

#Drop invalid hm_codes
print(len(delange_cd.index))
9457998
hm_code_mask = delange_cd.hm_code.ge(14) | delange_cd.hm_code.le(6)
hm_code_mask.sum()
1362922
delange_cd.drop(delange_cd[hm_code_mask].index, inplace=True)
print(len(delange_cd.index))
8095076

'''
delange_cd
           hm_variant_id      hm_rsid  hm_chrom    hm_pos hm_other_allele hm_effect_allele  hm_beta  hm_odds_ratio  ...  p_value  chromosome  base_pair_location  odds_ratio ci_lower ci_upper  effect_allele_frequency   variant_id\r
0         10_9958055_A_G    rs6602381        10   9958055               A                G   0.0109            NaN  ...  0.50440          10             9958055         NaN      NaN      NaN                      NaN    rs6602381\r
1        10_98240868_A_G    rs7899632        10  98240868               A                G   0.0181            NaN  ...  0.26050          10            98240868         NaN      NaN      NaN                      NaN    rs7899632\r
2        10_98240888_A_C   rs61875309        10  98240888               A                C  -0.0264            NaN  ...  0.18320          10            98240888         NaN      NaN      NaN                      NaN   rs61875309\r
3        10_98242110_C_T  rs150203744        10  98242110               C                T   0.0877            NaN  ...  0.22670          10            98242110         NaN      NaN      NaN                      NaN  rs150203744\r
4        10_98242707_T_C  rs111551711        10  98242707               T                C  -0.1536            NaN  ...  0.09955          10            98242707         NaN      NaN      NaN                      NaN  rs111551711\r
...                  ...          ...       ...       ...             ...              ...      ...            ...  ...      ...         ...                 ...         ...      ...      ...                      ...            ...
9469586   9_97234767_A_G   rs10817273         9  97234767               A                G   0.0005            NaN  ...  0.97600           9            97234767         NaN      NaN      NaN                      NaN   rs10817273\r
9469587   9_97235425_C_T   rs11794422         9  97235425               C                T   0.0083            NaN  ...  0.60850           9            97235425         NaN      NaN      NaN                      NaN   rs11794422\r
9469588   9_97236121_C_T   rs10981296         9  97236121               C                T   0.0071            NaN  ...  0.65960           9            97236121         NaN      NaN      NaN                      NaN   rs10981296\r
9469590   9_97236872_G_A   rs80110029         9  97236872               G                A  -0.2163            NaN  ...  0.18640           9            97236872         NaN      NaN      NaN                      NaN   rs80110029\r
9469591   9_97237186_A_G   rs10981301         9  97237186               A                G  -0.0171            NaN  ...  0.54390           9            97237186         NaN      NaN      NaN                      NaN   rs10981301\r

[8095076 rows x 24 columns]
'''
#need to type cast, etc.
remap_dict = {'X':23}
delange_cd['hm_chrom'] = delange_cd['hm_chrom'].replace(remap_dict)
delange_cd['hm_chrom'] = delange_cd['hm_chrom'].astype('int64')
delange_cd['hm_pos'] = delange_cd['hm_pos'].astype('int64')
delange_cd = delange_cd[delangecd_columns].sort_values(by=['hm_chrom', 'hm_pos'])

'''
delange_cd
           hm_variant_id      hm_rsid  hm_chrom    hm_pos hm_other_allele hm_effect_allele  hm_beta  hm_odds_ratio  hm_ci_lower  hm_ci_upper  p_value
3986546     1_778897_C_A  rs138660747         1    778897               C                A   0.0542            NaN          NaN          NaN  0.69360
3991597     1_796338_T_C   rs58276399         1    796338               T                C  -0.0583            NaN          NaN          NaN  0.05410
3991975     1_797429_T_C   rs12131618         1    797429               T                C   0.0384            NaN          NaN          NaN  0.32060
3992356     1_798969_T_C  rs141242758         1    798969               T                C  -0.0589            NaN          NaN          NaN  0.04911
3993411     1_801309_T_C  rs181876450         1    801309               T                C   0.0055            NaN          NaN          NaN  0.95820
...                  ...          ...       ...       ...             ...              ...      ...            ...          ...          ...      ...
4559464  22_50791377_T_C    rs9616985        22  50791377               T                C   0.0115            NaN          NaN          NaN  0.73410
4559465  22_50791427_G_A  rs144549712        22  50791427               G                A   0.0232            NaN          NaN          NaN  0.41290
4559466  22_50796371_G_A  rs191117135        22  50796371               G                A   0.1787            NaN          NaN          NaN  0.04211
4559467  22_50798635_T_C    rs3896457        22  50798635               T                C  -0.0011            NaN          NaN          NaN  0.95550
4559468  22_50799821_A_C  rs149733995        22  50799821               A                C  -0.0075            NaN          NaN          NaN  0.83210

[8095076 rows x 11 columns]
'''
delange_cd.equals(delange_cd.sort_values(by=['hm_chrom', 'hm_pos']))
oc_delange = ['hm_chrom', 'hm_pos','hm_other_allele', 'hm_effect_allele', 'hm_beta', 'p_value']

'''
delange_cd[oc_delange].to_csv('delange_cd_gwas_lz'+version+'.txt', sep='\t', index=False)
os.system("gzip -k " + 'delange_cd_gwas_lz'+version+'.txt &')
'''

'''
Study #16: Tsoi Psoriasis
http://ftp.ebi.ac.uk/pub/databases/gwas/summary_statistics/GCST005001-GCST006000/GCST005527/harmonised/23143594-GCST005527-EFO_0000676.h.tsv.gz
23143594
'''
TsoiPSO = sumstatspath + "23143594-GCST005527-EFO_0000676.h.tsv"
tsoi_pso = pd.read_csv(TsoiPSO, lineterminator='\n', sep='\t')
tsoi_pso.columns.to_list()

tsoi_pso.isna().sum()
pd.unique(tsoi_pso.hm_code)

'''
tsoi_pso.isna().sum()
hm_variant_id                   5729
hm_rsid                         5729
hm_chrom                        5729
hm_pos                          5729
hm_other_allele                 5729
hm_effect_allele                5729
hm_beta                        20948
hm_odds_ratio                  20471
hm_ci_lower                    20471
hm_ci_upper                    20471
hm_effect_allele_frequency    159609
hm_code                            0
chromosome                         0
base_pair_location                 0
variant_id                         0
other_allele                       0
effect_allele                      0
p_value                            0
beta                           19565
standard_error                 19565
odds_ratio                     19565
ci_lower                       19565
ci_upper                       19565
effect_allele_frequency\r          0
dtype: int64
>>> pd.unique(tsoi_pso.hm_code)
array([10, 11,  5,  6, 14, 15, 12, 13])
'''

tsoipso_columns = ['hm_variant_id', 'hm_rsid', 'hm_chrom', 'hm_pos', 'hm_other_allele', 'hm_effect_allele', 'hm_beta', 'hm_odds_ratio', 'hm_ci_lower', 'hm_ci_upper', 'p_value', 'standard_error']

'''
tsoi_pso
          hm_variant_id      hm_rsid  hm_chrom       hm_pos hm_other_allele hm_effect_allele   hm_beta  hm_odds_ratio  ...  effect_allele   p_value      beta  standard_error  odds_ratio    ci_lower            ci_upper effect_allele_frequency\r
0         10_251693_A_G    rs3125037      10.0     251693.0               A                G  0.039125       1.039900  ...              G  0.224159  0.039125        0.032187      1.0399    0.976323  1.1076169897933310                      NA\r
1         10_308361_C_T   rs35198327      10.0     308361.0               C                T -0.007174       0.992851  ...              C  0.982580  0.007174        0.328570      1.0072     0.52897  1.9177860613577884                      NA\r
2         10_318459_A_G    rs9804310      10.0     318459.0               A                G  0.040374       1.041200  ...              G  0.654816  0.040374        0.090305      1.0412    0.872297  1.2428071439949373                      NA\r
3         10_361644_G_A    rs3922851      10.0     361644.0               G                A -0.015825       0.984300  ...              A  0.466686 -0.015825        0.021741      0.9843    0.943239  1.0271489248740522                      NA\r
4         10_475246_C_T   rs11252630      10.0     475246.0               C                T  0.019018       1.019200  ...              T  0.400253  0.019018        0.022609      1.0192    0.975022  1.0653801033565744                      NA\r
...                 ...          ...       ...          ...             ...              ...       ...            ...  ...            ...       ...       ...             ...         ...         ...                 ...                       ...
159604  9_137397708_G_A   rs11523300       9.0  137397708.0               G                A -0.024703       0.975600  ...              A  0.486761 -0.024703        0.035519      0.9756    0.909991             1.04594                      NA\r
159605  9_137422632_C_A   rs13288824       9.0  137422632.0               C                A  0.022446       1.022700  ...              A  0.586795  0.022446        0.041300      1.0227    0.943176             1.10893                      NA\r
159606  9_137520818_G_A    rs2987631       9.0  137520818.0               G                A -0.416122       0.659600  ...              A  0.886616 -0.416122        2.918348      0.6596  0.00216338             201.108                      NA\r
159607  9_137613754_C_A  rs945589915       9.0  137613754.0               C                A       NaN            NaN  ...              A  0.605084       NaN             NaN         NaN         NaN                 NaN                      NA\r
159608  9_138096907_A_G    rs3812541       9.0  138096907.0               A                G  0.005783       1.005800  ...              G  0.480852  0.005783        0.008204      1.0058    0.989756              1.0221                      NA\r

[159609 rows x 24 columns]
'''

#Drop hm_variant_id NA
#no hm_odds_ratio that is not NaN
#DROP hm_beta NA
#do not drop hm_odds_ratio due to ... all of them are NaN
tsoi_pso.dropna(axis=0, subset=['hm_variant_id', 'hm_beta', 'hm_odds_ratio'], inplace=True)

'''
tsoi_pso
          hm_variant_id     hm_rsid  hm_chrom       hm_pos hm_other_allele hm_effect_allele   hm_beta  hm_odds_ratio  ...  effect_allele   p_value      beta  standard_error  odds_ratio    ci_lower            ci_upper effect_allele_frequency\r
0         10_251693_A_G   rs3125037      10.0     251693.0               A                G  0.039125       1.039900  ...              G  0.224159  0.039125        0.032187      1.0399    0.976323  1.1076169897933310                      NA\r
1         10_308361_C_T  rs35198327      10.0     308361.0               C                T -0.007174       0.992851  ...              C  0.982580  0.007174        0.328570      1.0072     0.52897  1.9177860613577884                      NA\r
2         10_318459_A_G   rs9804310      10.0     318459.0               A                G  0.040374       1.041200  ...              G  0.654816  0.040374        0.090305      1.0412    0.872297  1.2428071439949373                      NA\r
3         10_361644_G_A   rs3922851      10.0     361644.0               G                A -0.015825       0.984300  ...              A  0.466686 -0.015825        0.021741      0.9843    0.943239  1.0271489248740522                      NA\r
4         10_475246_C_T  rs11252630      10.0     475246.0               C                T  0.019018       1.019200  ...              T  0.400253  0.019018        0.022609      1.0192    0.975022  1.0653801033565744                      NA\r
...                 ...         ...       ...          ...             ...              ...       ...            ...  ...            ...       ...       ...             ...         ...         ...                 ...                       ...
159602  9_137245652_G_T  rs13300833       9.0  137245652.0               G                T -0.004008       0.996000  ...              T  0.988705 -0.004008        0.283115      0.9960     0.57183             1.73481                      NA\r
159604  9_137397708_G_A  rs11523300       9.0  137397708.0               G                A -0.024703       0.975600  ...              A  0.486761 -0.024703        0.035519      0.9756    0.909991             1.04594                      NA\r
159605  9_137422632_C_A  rs13288824       9.0  137422632.0               C                A  0.022446       1.022700  ...              A  0.586795  0.022446        0.041300      1.0227    0.943176             1.10893                      NA\r
159606  9_137520818_G_A   rs2987631       9.0  137520818.0               G                A -0.416122       0.659600  ...              A  0.886616 -0.416122        2.918348      0.6596  0.00216338             201.108                      NA\r
159608  9_138096907_A_G   rs3812541       9.0  138096907.0               A                G  0.005783       1.005800  ...              G  0.480852  0.005783        0.008204      1.0058    0.989756              1.0221                      NA\r

[138661 rows x 24 columns]
'''

#Drop invalid hm_codes
print(len(tsoi_pso.index))
138661
hm_code_mask = tsoi_pso.hm_code.ge(14) | tsoi_pso.hm_code.le(6)
hm_code_mask.sum()
16302
tsoi_pso.drop(tsoi_pso[hm_code_mask].index, inplace=True)
print(len(tsoi_pso.index))
122359

'''
tsoi_pso
          hm_variant_id     hm_rsid  hm_chrom       hm_pos hm_other_allele hm_effect_allele   hm_beta  hm_odds_ratio  ...  effect_allele   p_value      beta  standard_error  odds_ratio    ci_lower            ci_upper effect_allele_frequency\r
0         10_251693_A_G   rs3125037      10.0     251693.0               A                G  0.039125       1.039900  ...              G  0.224159  0.039125        0.032187      1.0399    0.976323  1.1076169897933310                      NA\r
1         10_308361_C_T  rs35198327      10.0     308361.0               C                T -0.007174       0.992851  ...              C  0.982580  0.007174        0.328570      1.0072     0.52897  1.9177860613577884                      NA\r
2         10_318459_A_G   rs9804310      10.0     318459.0               A                G  0.040374       1.041200  ...              G  0.654816  0.040374        0.090305      1.0412    0.872297  1.2428071439949373                      NA\r
3         10_361644_G_A   rs3922851      10.0     361644.0               G                A -0.015825       0.984300  ...              A  0.466686 -0.015825        0.021741      0.9843    0.943239  1.0271489248740522                      NA\r
4         10_475246_C_T  rs11252630      10.0     475246.0               C                T  0.019018       1.019200  ...              T  0.400253  0.019018        0.022609      1.0192    0.975022  1.0653801033565744                      NA\r
...                 ...         ...       ...          ...             ...              ...       ...            ...  ...            ...       ...       ...             ...         ...         ...                 ...                       ...
159602  9_137245652_G_T  rs13300833       9.0  137245652.0               G                T -0.004008       0.996000  ...              T  0.988705 -0.004008        0.283115      0.9960     0.57183             1.73481                      NA\r
159604  9_137397708_G_A  rs11523300       9.0  137397708.0               G                A -0.024703       0.975600  ...              A  0.486761 -0.024703        0.035519      0.9756    0.909991             1.04594                      NA\r
159605  9_137422632_C_A  rs13288824       9.0  137422632.0               C                A  0.022446       1.022700  ...              A  0.586795  0.022446        0.041300      1.0227    0.943176             1.10893                      NA\r
159606  9_137520818_G_A   rs2987631       9.0  137520818.0               G                A -0.416122       0.659600  ...              A  0.886616 -0.416122        2.918348      0.6596  0.00216338             201.108                      NA\r
159608  9_138096907_A_G   rs3812541       9.0  138096907.0               A                G  0.005783       1.005800  ...              G  0.480852  0.005783        0.008204      1.0058    0.989756              1.0221                      NA\r

[122359 rows x 24 columns]
'''
#need to type cast, etc.
remap_dict = {'X':23}
tsoi_pso['hm_chrom'] = tsoi_pso['hm_chrom'].replace(remap_dict)
tsoi_pso['hm_chrom'] = tsoi_pso['hm_chrom'].astype('int64')
tsoi_pso['hm_pos'] = tsoi_pso['hm_pos'].astype('int64')
tsoi_pso = tsoi_pso[tsoipso_columns].sort_values(by=['hm_chrom', 'hm_pos'])

'''
tsoi_pso
         hm_variant_id     hm_rsid  hm_chrom    hm_pos hm_other_allele hm_effect_allele   hm_beta  hm_odds_ratio  hm_ci_lower         hm_ci_upper   p_value
51272    1_1182895_C_T  rs61733845         1   1182895               C                T  0.157089         1.1701     1.028210  1.3315709454346485  0.017229
51273    1_1185051_G_A   rs1320571         1   1185051               G                A -0.020815         0.9794     0.903627  1.0615266163833759  0.612395
51274    1_1199862_A_C   rs9729550         1   1199862               A                C  0.003992         1.0040     0.892610  1.1292908398772332  0.946952
51275    1_1205055_G_T   rs1815606         1   1205055               G                T  0.007174         1.0072     0.928000   1.093158988658479  0.863677
51276    1_1228424_C_T   rs7515488         1   1228424               C                T  0.081949         1.0854     0.976021   1.207036719488366  0.130498
...                ...         ...       ...       ...             ...              ...       ...            ...          ...                 ...       ...
76985  22_50656498_C_T   rs9616810        22  50656498               C                T -0.005013         0.9950     0.843637             1.17352  0.952525
76986  22_50667128_C_T   rs9616812        22  50667128               C                T  0.017938         1.0181     0.883512             1.17319  0.804160
76987  22_50671564_T_C   rs9628185        22  50671564               T                C  0.018625         1.0188     0.878799              1.1811  0.804943
76988  22_50695758_A_G   rs8135777        22  50695758               A                G  0.484338         1.6231     0.381854             6.89912  0.511809
76989  22_50718238_C_T   rs9628187        22  50718238               C                T  0.018527         1.0187     0.868891             1.19434  0.819417

[122359 rows x 11 columns]
'''

tsoi_pso.equals(tsoi_pso.sort_values(by=['hm_chrom', 'hm_pos']))
oc_tsoi = ['hm_chrom', 'hm_pos','hm_other_allele', 'hm_effect_allele', 'hm_beta', 'p_value']

'''
tsoi_pso[oc_tsoi].to_csv('tsoi_pso_gwas_lz'+version+'.txt', sep='\t', index=False)
os.system("gzip -k " + 'tsoi_pso_gwas_lz'+version+'.txt &')
'''

'''
Study #17: Hinks JIA
http://ftp.ebi.ac.uk/pub/databases/gwas/summary_statistics/GCST005001-GCST006000/GCST005528/harmonised/23603761-GCST005528-EFO_1001999.h.tsv.gz
23603761
'''
HinksJIA = sumstatspath + "23603761-GCST005528-EFO_1001999.h.tsv"
hinks_jia = pd.read_csv(HinksJIA, lineterminator='\n', sep='\t')
hinks_jia.columns.to_list()

hinks_jia.isna().sum()
pd.unique(hinks_jia.hm_code)

'''
hinks_jia.isna().sum()
hm_variant_id                  18216
hm_rsid                        18216
hm_chrom                       18216
hm_pos                         18216
hm_other_allele                18216
hm_effect_allele               18216
hm_beta                        18473
hm_odds_ratio                  18216
hm_ci_lower                    18223
hm_ci_upper                    18223
hm_effect_allele_frequency    122240
hm_code                            0
chromosome                         0
base_pair_location                 0
variant_id                         0
other_allele                    3778
effect_allele                      0
p_value                            0
beta                               0
standard_error                     7
odds_ratio                         0
ci_lower                           7
ci_upper                           7
effect_allele_frequency\r          0
dtype: int64
>>> pd.unique(hinks_jia.hm_code)
array([10, 13, 12, 11,  9, 14, 15])
'''

hinksjia_columns = ['hm_variant_id', 'hm_rsid', 'hm_chrom', 'hm_pos', 'hm_other_allele', 'hm_effect_allele', 'hm_beta', 'hm_odds_ratio', 'hm_ci_lower', 'hm_ci_upper', 'p_value', 'standard_error']

'''
hinks_jia
        hm_variant_id     hm_rsid  hm_chrom    hm_pos hm_other_allele hm_effect_allele   hm_beta  hm_odds_ratio  ...  effect_allele  p_value      beta  standard_error  odds_ratio  ci_lower  ci_upper effect_allele_frequency\r
0       10_251693_A_G   rs3125037      10.0  251693.0               A                G -0.003205       0.996800  ...              G  0.95750 -0.003205        0.060144      0.9968  0.885957  1.121511                      NA\r
1       10_308361_C_T  rs35198327      10.0  308361.0               C                T  0.004912       1.004924  ...              G  0.92770 -0.004912        0.054134      0.9951  0.894926  1.106487                      NA\r
2       10_318459_A_G   rs9804310      10.0  318459.0               A                G  0.078811       1.082000  ...              G  0.28300  0.078811        0.073408      1.0820  0.937003  1.249434                      NA\r
3       10_361644_G_A   rs3922851      10.0  361644.0               G                A -0.054562       0.946900  ...              A  0.32730 -0.054562        0.055700      0.9469  0.848968  1.056129                      NA\r
4       10_475246_C_T  rs11252630      10.0  475246.0               C                T  0.028587       1.029000  ...              A  0.59330  0.028587        0.053529      1.0290  0.926512  1.142825                      NA\r
...               ...         ...       ...       ...             ...              ...       ...            ...  ...            ...      ...       ...             ...         ...       ...       ...                       ...
122235            NaN         NaN       NaN       NaN             NaN              NaN       NaN            NaN  ...              G  0.07698  0.101654        0.057481      1.1070  0.989052  1.239014                      NA\r
122236            NaN         NaN       NaN       NaN             NaN              NaN       NaN            NaN  ...              G  0.49870 -0.186691        0.275951      0.8297  0.483088  1.425002                      NA\r
122237            NaN         NaN       NaN       NaN             NaN              NaN       NaN            NaN  ...              A  0.69380  0.025668        0.065196      1.0260  0.902925  1.165851                      NA\r
122238            NaN         NaN       NaN       NaN             NaN              NaN       NaN            NaN  ...              C  0.07290  0.140631        0.078413      1.1510  0.987026  1.342215                      NA\r
122239            NaN         NaN       NaN       NaN             NaN              NaN       NaN            NaN  ...              G  0.70000 -0.021121        0.054815      0.9791  0.879361  1.090152                      NA\r

[122240 rows x 24 columns]
'''

#Drop hm_variant_id NA
#no hm_odds_ratio that is not NaN
#DROP hm_beta NA
#do not drop hm_odds_ratio due to ... all of them are NaN
hinks_jia.dropna(axis=0, subset=['hm_variant_id', 'hm_beta', 'hm_odds_ratio'], inplace=True)

'''
hinks_jia
          hm_variant_id     hm_rsid  hm_chrom       hm_pos hm_other_allele hm_effect_allele   hm_beta  hm_odds_ratio  ...  effect_allele  p_value      beta  standard_error  odds_ratio  ci_lower  ci_upper effect_allele_frequency\r
0         10_251693_A_G   rs3125037      10.0     251693.0               A                G -0.003205       0.996800  ...              G   0.9575 -0.003205        0.060144      0.9968  0.885957  1.121511                      NA\r
1         10_308361_C_T  rs35198327      10.0     308361.0               C                T  0.004912       1.004924  ...              G   0.9277 -0.004912        0.054134      0.9951  0.894926  1.106487                      NA\r
2         10_318459_A_G   rs9804310      10.0     318459.0               A                G  0.078811       1.082000  ...              G   0.2830  0.078811        0.073408      1.0820  0.937003  1.249434                      NA\r
3         10_361644_G_A   rs3922851      10.0     361644.0               G                A -0.054562       0.946900  ...              A   0.3273 -0.054562        0.055700      0.9469  0.848968  1.056129                      NA\r
4         10_475246_C_T  rs11252630      10.0     475246.0               C                T  0.028587       1.029000  ...              A   0.5933  0.028587        0.053529      1.0290  0.926512  1.142825                      NA\r
...                 ...         ...       ...          ...             ...              ...       ...            ...  ...            ...      ...       ...             ...         ...       ...       ...                       ...
120521  9_136916143_C_T   rs4880162       9.0  136916143.0               C                T -0.050693       0.950570  ...              G   0.4029  0.050693        0.060605      1.0520  0.934174  1.184687                      NA\r
120522  9_137142577_C_A   rs4880215       9.0  137142577.0               C                A -0.040405       0.960400  ...              A   0.4677 -0.040405        0.055637      0.9604  0.861177  1.071055                      NA\r
120523  9_137245305_G_A   rs9775264       9.0  137245305.0               G                A -0.006976       0.993049  ...              G   0.9787  0.006976        0.261271      1.0070  0.603436  1.680459                      NA\r
120525  9_137397708_G_A  rs11523300       9.0  137397708.0               G                A  0.088011       1.092000  ...              A   0.1570  0.088011        0.062188      1.0920  0.966689  1.233554                      NA\r
120526  9_138096907_A_G   rs3812541       9.0  138096907.0               A                G  0.044973       1.046000  ...              G   0.4968  0.044973        0.066183      1.0460  0.918747  1.190879                      NA\r

[103767 rows x 24 columns]
'''

#Drop invalid hm_codes
print(len(hinks_jia.index))
103767
hm_code_mask = hinks_jia.hm_code.ge(14) | hinks_jia.hm_code.le(6)
hm_code_mask.sum()
0
hinks_jia.drop(hinks_jia[hm_code_mask].index, inplace=True)
print(len(hinks_jia.index))
103767

'''
hinks_jia
          hm_variant_id     hm_rsid  hm_chrom       hm_pos hm_other_allele hm_effect_allele   hm_beta  hm_odds_ratio  ...  effect_allele  p_value      beta  standard_error  odds_ratio  ci_lower  ci_upper effect_allele_frequency\r
0         10_251693_A_G   rs3125037      10.0     251693.0               A                G -0.003205       0.996800  ...              G   0.9575 -0.003205        0.060144      0.9968  0.885957  1.121511                      NA\r
1         10_308361_C_T  rs35198327      10.0     308361.0               C                T  0.004912       1.004924  ...              G   0.9277 -0.004912        0.054134      0.9951  0.894926  1.106487                      NA\r
2         10_318459_A_G   rs9804310      10.0     318459.0               A                G  0.078811       1.082000  ...              G   0.2830  0.078811        0.073408      1.0820  0.937003  1.249434                      NA\r
3         10_361644_G_A   rs3922851      10.0     361644.0               G                A -0.054562       0.946900  ...              A   0.3273 -0.054562        0.055700      0.9469  0.848968  1.056129                      NA\r
4         10_475246_C_T  rs11252630      10.0     475246.0               C                T  0.028587       1.029000  ...              A   0.5933  0.028587        0.053529      1.0290  0.926512  1.142825                      NA\r
...                 ...         ...       ...          ...             ...              ...       ...            ...  ...            ...      ...       ...             ...         ...       ...       ...                       ...
120521  9_136916143_C_T   rs4880162       9.0  136916143.0               C                T -0.050693       0.950570  ...              G   0.4029  0.050693        0.060605      1.0520  0.934174  1.184687                      NA\r
120522  9_137142577_C_A   rs4880215       9.0  137142577.0               C                A -0.040405       0.960400  ...              A   0.4677 -0.040405        0.055637      0.9604  0.861177  1.071055                      NA\r
120523  9_137245305_G_A   rs9775264       9.0  137245305.0               G                A -0.006976       0.993049  ...              G   0.9787  0.006976        0.261271      1.0070  0.603436  1.680459                      NA\r
120525  9_137397708_G_A  rs11523300       9.0  137397708.0               G                A  0.088011       1.092000  ...              A   0.1570  0.088011        0.062188      1.0920  0.966689  1.233554                      NA\r
120526  9_138096907_A_G   rs3812541       9.0  138096907.0               A                G  0.044973       1.046000  ...              G   0.4968  0.044973        0.066183      1.0460  0.918747  1.190879                      NA\r

[103767 rows x 24 columns]
'''
#need to type cast, etc.
remap_dict = {'X':23}
hinks_jia['hm_chrom'] = hinks_jia['hm_chrom'].replace(remap_dict)
hinks_jia['hm_chrom'] = hinks_jia['hm_chrom'].astype('int64')
hinks_jia['hm_pos'] = hinks_jia['hm_pos'].astype('int64')
hinks_jia = hinks_jia[hinksjia_columns].sort_values(by=['hm_chrom', 'hm_pos'])

'''
hinks_jia
         hm_variant_id     hm_rsid  hm_chrom    hm_pos hm_other_allele hm_effect_allele   hm_beta  hm_odds_ratio  hm_ci_lower  hm_ci_upper  p_value
39022    1_1182895_C_T  rs61733845         1   1182895               C                T -0.168537       0.844900     0.642016     1.111897   0.2290
39023    1_1185051_G_A   rs1320571         1   1185051               G                A  0.043059       1.044000     0.812166     1.342012   0.7368
39024    1_1199862_A_C   rs9729550         1   1199862               A                C -0.079693       0.923400     0.818482     1.041767   0.1953
39025    1_1205055_G_T   rs1815606         1   1205055               G                T -0.068065       0.934200     0.833952     1.046499   0.2399
39026    1_1228424_C_T   rs7515488         1   1228424               C                T  0.042101       1.043000     0.904489     1.202722   0.5625
...                ...         ...       ...       ...             ...              ...       ...            ...          ...          ...      ...
57483  22_50560753_C_T    rs140518        22  50560753               C                T -0.055435       0.946074     0.845536     1.058566   0.3335
57485  22_50656498_C_T   rs9616810        22  50656498               C                T  0.051643       1.053000     0.928988     1.193567   0.4192
57486  22_50667128_C_T   rs9616812        22  50667128               C                T -0.043221       0.957700     0.861691     1.064406   0.4226
57487  22_50671564_T_C   rs9628185        22  50671564               T                C -0.041343       0.959500     0.863338     1.066373   0.4429
57488  22_50718238_C_T   rs9628187        22  50718238               C                T -0.030459       0.970000     0.851943     1.104416   0.6455

[103767 rows x 11 columns]
'''

hinks_jia.equals(tsoi_pso.sort_values(by=['hm_chrom', 'hm_pos']))
oc_hinks = ['hm_chrom', 'hm_pos','hm_other_allele', 'hm_effect_allele', 'hm_beta', 'p_value']

'''
hinks_jia[oc_hinks].to_csv('hinks_jia_gwas_lz'+version+'.txt', sep='\t', index=False)
os.system("gzip -k " + 'hinks_jia_gwas_lz'+version+'.txt &')
'''

'''
Study #18: Beecham Multiple Sclerosis
http://ftp.ebi.ac.uk/pub/databases/gwas/summary_statistics/GCST005001-GCST006000/GCST005531/harmonised/24076602-GCST005531-EFO_0003885.h.tsv.gz
24076602
'''
BeechamMS = sumstatspath + "24076602-GCST005531-EFO_0003885.h.tsv"
beecham_ms = pd.read_csv(BeechamMS, lineterminator='\n', sep='\t')
beecham_ms.columns.to_list()

beecham_ms.isna().sum()
pd.unique(beecham_ms.hm_code)

'''
beecham_ms.isna().sum()
hm_variant_id                  20134
hm_rsid                        20134
hm_chrom                       20134
hm_pos                         20134
hm_other_allele                20134
hm_effect_allele               20134
hm_beta                        23160
hm_odds_ratio                  21667
hm_ci_lower                    21738
hm_ci_upper                    21738
hm_effect_allele_frequency    155249
hm_code                            0
chromosome                         0
base_pair_location                 0
variant_id                         0
other_allele                       0
effect_allele                      0
p_value                            0
beta                            1926
standard_error                  2010
odds_ratio                      1926
ci_lower                        2010
ci_upper                        2010
effect_allele_frequency\r          0
dtype: int64
>>> pd.unique(beecham_ms.hm_code)
array([10, 13, 12, 11,  9, 14, 15])
'''

beechamms_columns = ['hm_variant_id', 'hm_rsid', 'hm_chrom', 'hm_pos', 'hm_other_allele', 'hm_effect_allele', 'hm_beta', 'hm_odds_ratio', 'hm_ci_lower', 'hm_ci_upper', 'p_value', 'standard_error']

'''
beecham_ms
          hm_variant_id     hm_rsid  hm_chrom       hm_pos hm_other_allele hm_effect_allele   hm_beta  ...  p_value      beta standard_error  odds_ratio             ci_lower            ci_upper  effect_allele_frequency\r
0         10_251693_A_G   rs3125037      10.0     251693.0               A                G  0.026642  ...   0.1540  0.026642       0.018689      1.0270  0.99006129422610567  1.0653168709362004                       NA\r
1         10_308361_C_T  rs35198327      10.0     308361.0               C                T -0.014889  ...   0.3660  0.014889       0.016470      1.0150  0.98275805610786704  1.0482997250413002                       NA\r
2         10_318459_A_G   rs9804310      10.0     318459.0               A                G  0.005982  ...   0.7990  0.005982       0.023492      1.0060  0.96072938601888036  1.0534038145681446                       NA\r
3         10_361644_G_A   rs3922851      10.0     361644.0               G                A  0.010940  ...   0.5170  0.010940       0.016883      1.0110  0.97809234580754156  1.0450148233765257                       NA\r
4         10_475246_C_T  rs11252630      10.0     475246.0               C                T -0.011971  ...   0.4590 -0.011971       0.016167      0.9881  0.95728114313844759  1.0199110438956966                       NA\r
...                 ...         ...       ...          ...             ...              ...       ...  ...      ...       ...            ...         ...                  ...                 ...                        ...
155244  9_137397708_G_A  rs11523300       9.0  137397708.0               G                A  0.035367  ...   0.0661  0.035367       0.019245      1.0360  0.99764998389920501  1.0758242042014985                       NA\r
155245  9_137422632_C_A  rs13288824       9.0  137422632.0               C                A  0.015873  ...   0.3710  0.015873       0.017743      1.0160  0.98127368514967139  1.0519552451287353                       NA\r
155246  9_137520818_G_A   rs2987631       9.0  137520818.0               G                A -0.396159  ...   0.4960 -0.396159       0.581904      0.6729  0.21509173389519762  2.1051223206030865                       NA\r
155247  9_138096907_A_G   rs3812541       9.0  138096907.0               A                G -0.001001  ...   0.9690 -0.001001       0.025745      0.9990  0.94984170587998007  1.0507024421247146                       NA\r
155248  9_138137446_T_G   rs9695626       9.0  138137446.0               T                G -0.058269  ...   0.2850  0.058269       0.054500      1.0600  0.95260870076270698  1.1794979398155705                       NA\r

[155249 rows x 24 columns]
'''

#Drop hm_variant_id NA
#no hm_odds_ratio that is not NaN
#DROP hm_beta NA
#do not drop hm_odds_ratio due to ... all of them are NaN
beecham_ms.dropna(axis=0, subset=['hm_variant_id', 'hm_beta', 'hm_odds_ratio'], inplace=True)

'''
beecham_ms
          hm_variant_id     hm_rsid  hm_chrom       hm_pos hm_other_allele hm_effect_allele   hm_beta  ...  p_value      beta standard_error  odds_ratio             ci_lower            ci_upper  effect_allele_frequency\r
0         10_251693_A_G   rs3125037      10.0     251693.0               A                G  0.026642  ...   0.1540  0.026642       0.018689      1.0270  0.99006129422610567  1.0653168709362004                       NA\r
1         10_308361_C_T  rs35198327      10.0     308361.0               C                T -0.014889  ...   0.3660  0.014889       0.016470      1.0150  0.98275805610786704  1.0482997250413002                       NA\r
2         10_318459_A_G   rs9804310      10.0     318459.0               A                G  0.005982  ...   0.7990  0.005982       0.023492      1.0060  0.96072938601888036  1.0534038145681446                       NA\r
3         10_361644_G_A   rs3922851      10.0     361644.0               G                A  0.010940  ...   0.5170  0.010940       0.016883      1.0110  0.97809234580754156  1.0450148233765257                       NA\r
4         10_475246_C_T  rs11252630      10.0     475246.0               C                T -0.011971  ...   0.4590 -0.011971       0.016167      0.9881  0.95728114313844759  1.0199110438956966                       NA\r
...                 ...         ...       ...          ...             ...              ...       ...  ...      ...       ...            ...         ...                  ...                 ...                        ...
155244  9_137397708_G_A  rs11523300       9.0  137397708.0               G                A  0.035367  ...   0.0661  0.035367       0.019245      1.0360  0.99764998389920501  1.0758242042014985                       NA\r
155245  9_137422632_C_A  rs13288824       9.0  137422632.0               C                A  0.015873  ...   0.3710  0.015873       0.017743      1.0160  0.98127368514967139  1.0519552451287353                       NA\r
155246  9_137520818_G_A   rs2987631       9.0  137520818.0               G                A -0.396159  ...   0.4960 -0.396159       0.581904      0.6729  0.21509173389519762  2.1051223206030865                       NA\r
155247  9_138096907_A_G   rs3812541       9.0  138096907.0               A                G -0.001001  ...   0.9690 -0.001001       0.025745      0.9990  0.94984170587998007  1.0507024421247146                       NA\r
155248  9_138137446_T_G   rs9695626       9.0  138137446.0               T                G -0.058269  ...   0.2850  0.058269       0.054500      1.0600  0.95260870076270698  1.1794979398155705                       NA\r

[132089 rows x 24 columns]
'''

#Drop invalid hm_codes
print(len(beecham_ms.index))
132089
hm_code_mask = beecham_ms.hm_code.ge(14) | beecham_ms.hm_code.le(6)
hm_code_mask.sum()
0
beecham_ms.drop(beecham_ms[hm_code_mask].index, inplace=True)
print(len(beecham_ms.index))
132089

'''
 beecham_ms
          hm_variant_id     hm_rsid  hm_chrom       hm_pos hm_other_allele hm_effect_allele   hm_beta  ...  p_value      beta standard_error  odds_ratio             ci_lower            ci_upper  effect_allele_frequency\r
0         10_251693_A_G   rs3125037      10.0     251693.0               A                G  0.026642  ...   0.1540  0.026642       0.018689      1.0270  0.99006129422610567  1.0653168709362004                       NA\r
1         10_308361_C_T  rs35198327      10.0     308361.0               C                T -0.014889  ...   0.3660  0.014889       0.016470      1.0150  0.98275805610786704  1.0482997250413002                       NA\r
2         10_318459_A_G   rs9804310      10.0     318459.0               A                G  0.005982  ...   0.7990  0.005982       0.023492      1.0060  0.96072938601888036  1.0534038145681446                       NA\r
3         10_361644_G_A   rs3922851      10.0     361644.0               G                A  0.010940  ...   0.5170  0.010940       0.016883      1.0110  0.97809234580754156  1.0450148233765257                       NA\r
4         10_475246_C_T  rs11252630      10.0     475246.0               C                T -0.011971  ...   0.4590 -0.011971       0.016167      0.9881  0.95728114313844759  1.0199110438956966                       NA\r
...                 ...         ...       ...          ...             ...              ...       ...  ...      ...       ...            ...         ...                  ...                 ...                        ...
155244  9_137397708_G_A  rs11523300       9.0  137397708.0               G                A  0.035367  ...   0.0661  0.035367       0.019245      1.0360  0.99764998389920501  1.0758242042014985                       NA\r
155245  9_137422632_C_A  rs13288824       9.0  137422632.0               C                A  0.015873  ...   0.3710  0.015873       0.017743      1.0160  0.98127368514967139  1.0519552451287353                       NA\r
155246  9_137520818_G_A   rs2987631       9.0  137520818.0               G                A -0.396159  ...   0.4960 -0.396159       0.581904      0.6729  0.21509173389519762  2.1051223206030865                       NA\r
155247  9_138096907_A_G   rs3812541       9.0  138096907.0               A                G -0.001001  ...   0.9690 -0.001001       0.025745      0.9990  0.94984170587998007  1.0507024421247146                       NA\r
155248  9_138137446_T_G   rs9695626       9.0  138137446.0               T                G -0.058269  ...   0.2850  0.058269       0.054500      1.0600  0.95260870076270698  1.1794979398155705                       NA\r

[132089 rows x 24 columns]
'''
#need to type cast, etc.
remap_dict = {'X':23}
beecham_ms['hm_chrom'] = beecham_ms['hm_chrom'].replace(remap_dict)
beecham_ms['hm_chrom'] = beecham_ms['hm_chrom'].astype('int64')
beecham_ms['hm_pos'] = beecham_ms['hm_pos'].astype('int64')
beecham_ms = beecham_ms[beechamms_columns].sort_values(by=['hm_chrom', 'hm_pos'])

'''
beecham_ms\
...
         hm_variant_id     hm_rsid  hm_chrom    hm_pos hm_other_allele hm_effect_allele   hm_beta  hm_odds_ratio  hm_ci_lower         hm_ci_upper  p_value
49967    1_1182895_C_T  rs61733845         1   1182895               C                T -0.011971         0.9881     0.914437  1.0676971763025762    0.762
49968    1_1185051_G_A   rs1320571         1   1185051               G                A  0.025668         1.0260     0.950742  1.1072146259925333    0.509
49969    1_1199862_A_C   rs9729550         1   1199862               A                C  0.001000         1.0010     0.953293  1.0510949308412958    0.968
49970    1_1205055_G_T   rs1815606         1   1205055               G                T  0.002996         1.0030     0.966544  1.0408306968055108    0.874
49971    1_1228424_C_T   rs7515488         1   1228424               C                T  0.019803         1.0200     0.975796   1.066206563525329    0.381
...                ...         ...       ...       ...             ...              ...       ...            ...          ...                 ...      ...
74524  22_50656498_C_T   rs9616810        22  50656498               C                T  0.006976         1.0070     0.968181  1.0473755842106767    0.728
74525  22_50667128_C_T   rs9616812        22  50667128               C                T -0.002002         0.9980     0.971055  1.0256922064673688    0.886
74526  22_50671564_T_C   rs9628185        22  50671564               T                C  0.001000         1.0010     0.968947  1.0341127907285765    0.952
74527  22_50695758_A_G   rs8135777        22  50695758               A                G  0.631804         1.8810     0.780813   4.531379050901968    0.159
74528  22_50718238_C_T   rs9628187        22  50718238               C                T -0.006018         0.9940     0.952710  1.0370799724102215    0.781

[132089 rows x 11 columns]
'''

beecham_ms.equals(beecham_ms.sort_values(by=['hm_chrom', 'hm_pos']))
oc_beecham = ['hm_chrom', 'hm_pos','hm_other_allele', 'hm_effect_allele', 'hm_beta', 'p_value']

'''
beecham_ms[oc_beecham].to_csv('beecham_ms_gwas_lz'+version+'.txt', sep='\t', index=False)
os.system("gzip -k " + 'beecham_ms_gwas_lz'+version+'.txt &')
'''

'''
Study #19: Liu Primary Biliary Cirrhosis
http://ftp.ebi.ac.uk/pub/databases/gwas/summary_statistics/GCST005001-GCST006000/GCST005581/harmonised/22961000-GCST005581-EFO_1001486.h.tsv.gz
22961000
'''
LiuPBC = sumstatspath + "22961000-GCST005581-EFO_1001486.h.tsv"
liu_pbc = pd.read_csv(LiuPBC, lineterminator='\n', sep='\t')
liu_pbc.columns.to_list()

liu_pbc.isna().sum()
pd.unique(liu_pbc.hm_code)

'''
liu_pbc.isna().sum()
hm_variant_id                  20134
hm_rsid                        20134
hm_chrom                       20134
hm_pos                         20134
hm_other_allele                20134
hm_effect_allele               20134
hm_beta                        23160
hm_odds_ratio                  21667
hm_ci_lower                    21738
hm_ci_upper                    21738
hm_effect_allele_frequency    155249
hm_code                            0
chromosome                         0
base_pair_location                 0
variant_id                         0
other_allele                       0
effect_allele                      0
p_value                            0
beta                            1926
standard_error                  2010
odds_ratio                      1926
ci_lower                        2010
ci_upper                        2010
effect_allele_frequency\r          0
dtype: int64
>>> pd.unique(liu_pbc.hm_code)
array([10, 13, 12, 11,  9, 14, 15])
'''

liupbc_columns = ['hm_variant_id', 'hm_rsid', 'hm_chrom', 'hm_pos', 'hm_other_allele', 'hm_effect_allele', 'hm_beta', 'hm_odds_ratio', 'hm_ci_lower', 'hm_ci_upper', 'p_value', 'standard_error']

'''
liu_pbc
          hm_variant_id     hm_rsid  hm_chrom       hm_pos hm_other_allele hm_effect_allele   hm_beta  ...  p_value      beta standard_error  odds_ratio             ci_lower            ci_upper  effect_allele_frequency\r
0         10_251693_A_G   rs3125037      10.0     251693.0               A                G  0.026642  ...   0.1540  0.026642       0.018689      1.0270  0.99006129422610567  1.0653168709362004                       NA\r
1         10_308361_C_T  rs35198327      10.0     308361.0               C                T -0.014889  ...   0.3660  0.014889       0.016470      1.0150  0.98275805610786704  1.0482997250413002                       NA\r
2         10_318459_A_G   rs9804310      10.0     318459.0               A                G  0.005982  ...   0.7990  0.005982       0.023492      1.0060  0.96072938601888036  1.0534038145681446                       NA\r
3         10_361644_G_A   rs3922851      10.0     361644.0               G                A  0.010940  ...   0.5170  0.010940       0.016883      1.0110  0.97809234580754156  1.0450148233765257                       NA\r
4         10_475246_C_T  rs11252630      10.0     475246.0               C                T -0.011971  ...   0.4590 -0.011971       0.016167      0.9881  0.95728114313844759  1.0199110438956966                       NA\r
...                 ...         ...       ...          ...             ...              ...       ...  ...      ...       ...            ...         ...                  ...                 ...                        ...
155244  9_137397708_G_A  rs11523300       9.0  137397708.0               G                A  0.035367  ...   0.0661  0.035367       0.019245      1.0360  0.99764998389920501  1.0758242042014985                       NA\r
155245  9_137422632_C_A  rs13288824       9.0  137422632.0               C                A  0.015873  ...   0.3710  0.015873       0.017743      1.0160  0.98127368514967139  1.0519552451287353                       NA\r
155246  9_137520818_G_A   rs2987631       9.0  137520818.0               G                A -0.396159  ...   0.4960 -0.396159       0.581904      0.6729  0.21509173389519762  2.1051223206030865                       NA\r
155247  9_138096907_A_G   rs3812541       9.0  138096907.0               A                G -0.001001  ...   0.9690 -0.001001       0.025745      0.9990  0.94984170587998007  1.0507024421247146                       NA\r
155248  9_138137446_T_G   rs9695626       9.0  138137446.0               T                G -0.058269  ...   0.2850  0.058269       0.054500      1.0600  0.95260870076270698  1.1794979398155705                       NA\r

[155249 rows x 24 columns]
'''

#Drop hm_variant_id NA
#no hm_odds_ratio that is not NaN
#DROP hm_beta NA
#do not drop hm_odds_ratio due to ... all of them are NaN
liu_pbc.dropna(axis=0, subset=['hm_variant_id', 'hm_beta', 'hm_odds_ratio'], inplace=True)

'''
liu_pbc
          hm_variant_id     hm_rsid  hm_chrom       hm_pos hm_other_allele hm_effect_allele   hm_beta  ...  p_value      beta standard_error  odds_ratio             ci_lower            ci_upper  effect_allele_frequency\r
0         10_251693_A_G   rs3125037      10.0     251693.0               A                G  0.026642  ...   0.1540  0.026642       0.018689      1.0270  0.99006129422610567  1.0653168709362004                       NA\r
1         10_308361_C_T  rs35198327      10.0     308361.0               C                T -0.014889  ...   0.3660  0.014889       0.016470      1.0150  0.98275805610786704  1.0482997250413002                       NA\r
2         10_318459_A_G   rs9804310      10.0     318459.0               A                G  0.005982  ...   0.7990  0.005982       0.023492      1.0060  0.96072938601888036  1.0534038145681446                       NA\r
3         10_361644_G_A   rs3922851      10.0     361644.0               G                A  0.010940  ...   0.5170  0.010940       0.016883      1.0110  0.97809234580754156  1.0450148233765257                       NA\r
4         10_475246_C_T  rs11252630      10.0     475246.0               C                T -0.011971  ...   0.4590 -0.011971       0.016167      0.9881  0.95728114313844759  1.0199110438956966                       NA\r
...                 ...         ...       ...          ...             ...              ...       ...  ...      ...       ...            ...         ...                  ...                 ...                        ...
155244  9_137397708_G_A  rs11523300       9.0  137397708.0               G                A  0.035367  ...   0.0661  0.035367       0.019245      1.0360  0.99764998389920501  1.0758242042014985                       NA\r
155245  9_137422632_C_A  rs13288824       9.0  137422632.0               C                A  0.015873  ...   0.3710  0.015873       0.017743      1.0160  0.98127368514967139  1.0519552451287353                       NA\r
155246  9_137520818_G_A   rs2987631       9.0  137520818.0               G                A -0.396159  ...   0.4960 -0.396159       0.581904      0.6729  0.21509173389519762  2.1051223206030865                       NA\r
155247  9_138096907_A_G   rs3812541       9.0  138096907.0               A                G -0.001001  ...   0.9690 -0.001001       0.025745      0.9990  0.94984170587998007  1.0507024421247146                       NA\r
155248  9_138137446_T_G   rs9695626       9.0  138137446.0               T                G -0.058269  ...   0.2850  0.058269       0.054500      1.0600  0.95260870076270698  1.1794979398155705                       NA\r

[132089 rows x 24 columns]
'''

#Drop invalid hm_codes
print(len(liu_pbc.index))
132089
hm_code_mask = liu_pbc.hm_code.ge(14) | liu_pbc.hm_code.le(6)
hm_code_mask.sum()
0
liu_pbc.drop(liu_pbc[hm_code_mask].index, inplace=True)
print(len(liu_pbc.index))
132089

'''
liu_pbc
          hm_variant_id     hm_rsid  hm_chrom       hm_pos hm_other_allele hm_effect_allele   hm_beta  ...  p_value      beta standard_error  odds_ratio             ci_lower            ci_upper  effect_allele_frequency\r
0         10_251693_A_G   rs3125037      10.0     251693.0               A                G  0.026642  ...   0.1540  0.026642       0.018689      1.0270  0.99006129422610567  1.0653168709362004                       NA\r
1         10_308361_C_T  rs35198327      10.0     308361.0               C                T -0.014889  ...   0.3660  0.014889       0.016470      1.0150  0.98275805610786704  1.0482997250413002                       NA\r
2         10_318459_A_G   rs9804310      10.0     318459.0               A                G  0.005982  ...   0.7990  0.005982       0.023492      1.0060  0.96072938601888036  1.0534038145681446                       NA\r
3         10_361644_G_A   rs3922851      10.0     361644.0               G                A  0.010940  ...   0.5170  0.010940       0.016883      1.0110  0.97809234580754156  1.0450148233765257                       NA\r
4         10_475246_C_T  rs11252630      10.0     475246.0               C                T -0.011971  ...   0.4590 -0.011971       0.016167      0.9881  0.95728114313844759  1.0199110438956966                       NA\r
...                 ...         ...       ...          ...             ...              ...       ...  ...      ...       ...            ...         ...                  ...                 ...                        ...
155244  9_137397708_G_A  rs11523300       9.0  137397708.0               G                A  0.035367  ...   0.0661  0.035367       0.019245      1.0360  0.99764998389920501  1.0758242042014985                       NA\r
155245  9_137422632_C_A  rs13288824       9.0  137422632.0               C                A  0.015873  ...   0.3710  0.015873       0.017743      1.0160  0.98127368514967139  1.0519552451287353                       NA\r
155246  9_137520818_G_A   rs2987631       9.0  137520818.0               G                A -0.396159  ...   0.4960 -0.396159       0.581904      0.6729  0.21509173389519762  2.1051223206030865                       NA\r
155247  9_138096907_A_G   rs3812541       9.0  138096907.0               A                G -0.001001  ...   0.9690 -0.001001       0.025745      0.9990  0.94984170587998007  1.0507024421247146                       NA\r
155248  9_138137446_T_G   rs9695626       9.0  138137446.0               T                G -0.058269  ...   0.2850  0.058269       0.054500      1.0600  0.95260870076270698  1.1794979398155705                       NA\r

[132089 rows x 24 columns]
'''
#need to type cast, etc.
remap_dict = {'X':23}
liu_pbc['hm_chrom'] = liu_pbc['hm_chrom'].replace(remap_dict)
liu_pbc['hm_chrom'] = liu_pbc['hm_chrom'].astype('int64')
liu_pbc['hm_pos'] = liu_pbc['hm_pos'].astype('int64')
liu_pbc = liu_pbc[liupbc_columns].sort_values(by=['hm_chrom', 'hm_pos'])

'''
liu_pbc
         hm_variant_id     hm_rsid  hm_chrom    hm_pos hm_other_allele hm_effect_allele   hm_beta  hm_odds_ratio  hm_ci_lower         hm_ci_upper  p_value
49967    1_1182895_C_T  rs61733845         1   1182895               C                T -0.011971         0.9881     0.914437  1.0676971763025762    0.762
49968    1_1185051_G_A   rs1320571         1   1185051               G                A  0.025668         1.0260     0.950742  1.1072146259925333    0.509
49969    1_1199862_A_C   rs9729550         1   1199862               A                C  0.001000         1.0010     0.953293  1.0510949308412958    0.968
49970    1_1205055_G_T   rs1815606         1   1205055               G                T  0.002996         1.0030     0.966544  1.0408306968055108    0.874
49971    1_1228424_C_T   rs7515488         1   1228424               C                T  0.019803         1.0200     0.975796   1.066206563525329    0.381
...                ...         ...       ...       ...             ...              ...       ...            ...          ...                 ...      ...
74524  22_50656498_C_T   rs9616810        22  50656498               C                T  0.006976         1.0070     0.968181  1.0473755842106767    0.728
74525  22_50667128_C_T   rs9616812        22  50667128               C                T -0.002002         0.9980     0.971055  1.0256922064673688    0.886
74526  22_50671564_T_C   rs9628185        22  50671564               T                C  0.001000         1.0010     0.968947  1.0341127907285765    0.952
74527  22_50695758_A_G   rs8135777        22  50695758               A                G  0.631804         1.8810     0.780813   4.531379050901968    0.159
74528  22_50718238_C_T   rs9628187        22  50718238               C                T -0.006018         0.9940     0.952710  1.0370799724102215    0.781

[132089 rows x 11 columns]
'''

liu_pbc.equals(liu_pbc.sort_values(by=['hm_chrom', 'hm_pos']))
oc_liu = ['hm_chrom', 'hm_pos','hm_other_allele', 'hm_effect_allele', 'hm_beta', 'p_value']

'''
liu_pbc[oc_liu].to_csv('liu_pbc_gwas_lz'+version+'.txt', sep='\t', index=False)
os.system("gzip -k " + 'liu_pbc_gwas_lz'+version+'.txt &')
'''

'''
Study #20: Chen HgbA1C
http://ftp.ebi.ac.uk/pub/databases/gwas/summary_statistics/GCST90002001-GCST90003000/GCST90002244/harmonised/34059833-GCST90002244-EFO_0004541.h.tsv.gz
34059833
'''
ChenA1C = sumstatspath + "34059833-GCST90002244-EFO_0004541.h.tsv"
chen_a1c = pd.read_csv(ChenA1C, lineterminator='\n', sep='\t')
chen_a1c.columns.to_list()

chen_a1c.isna().sum()
pd.unique(chen_a1c.hm_code)

'''
chen_a1c.isna().sum()
hm_variant_id                        0
hm_rsid                              0
hm_chrom                             0
hm_pos                               0
hm_other_allele                      0
hm_effect_allele                     0
hm_beta                         162475
hm_odds_ratio                 30817670
hm_ci_lower                   30817670
hm_ci_upper                   30817670
hm_effect_allele_frequency    15636867
hm_code                              0
chromosome                           0
base_pair_location                   0
effect_allele                        0
other_allele                         0
effect_allele_frequency       15636867
beta                                 0
standard_error                       0
p_value                              0
odds_ratio                    30817670
ci_lower                      30817670
ci_upper                      30817670
variant_id\r                         0
dtype: int64
>>> pd.unique(chen_a1c.hm_code)
array([10,  6, 11,  5, 13, 12])
'''

chena1c_columns = ['hm_variant_id', 'hm_rsid', 'hm_chrom', 'hm_pos', 'hm_other_allele', 'hm_effect_allele', 'hm_beta', 'hm_odds_ratio', 'hm_ci_lower', 'hm_ci_upper', 'p_value', 'standard_error']

'''
chen_a1c
            hm_variant_id      hm_rsid hm_chrom     hm_pos hm_other_allele hm_effect_allele  hm_beta  hm_odds_ratio  ...  effect_allele_frequency    beta  standard_error  p_value  odds_ratio  ci_lower ci_upper   variant_id\r
0          1_99534456_G_T   rs10875231        1   99534456               G                T   0.0011            NaN  ...                    0.251  0.0011          0.0015  0.70520         NaN       NaN      NaN   rs10875231\r
1           1_9939948_G_A  rs186077422        1    9939948               G                A   0.0136            NaN  ...                    0.008  0.0136          0.0322  0.71810         NaN       NaN      NaN  rs186077422\r
2          1_99534579_A_T  rs114947036        1   99534579               A                T   0.0160            NaN  ...                    0.999 -0.0160          0.0895  0.62260         NaN       NaN      NaN  rs114947036\r
3          1_99535129_T_C  rs140126637        1   99535129               T                C  -0.1146            NaN  ...                    0.999  0.1146          0.0807  0.08056         NaN       NaN      NaN  rs140126637\r
4          1_99535213_T_C  rs543738424        1   99535213               T                C   0.0719            NaN  ...                      NaN -0.0719          0.1649  0.73750         NaN       NaN      NaN  rs543738424\r
...                   ...          ...      ...        ...             ...              ...      ...            ...  ...                      ...     ...             ...      ...         ...       ...      ...            ...
30817665  X_100744278_G_A  rs140486085        X  100744278               G                A  -0.0027            NaN  ...                      NaN -0.0027          0.1321  0.93150         NaN       NaN      NaN  rs140486085\r
30817666  X_100744360_G_A    rs2021704        X  100744360               G                A   0.0018            NaN  ...                      NaN  0.0018          0.0018  0.17050         NaN       NaN      NaN    rs2021704\r
30817667  X_100744400_A_G  rs146574849        X  100744400               A                G  -0.0048            NaN  ...                      NaN  0.0048          0.0026  0.06514         NaN       NaN      NaN  rs146574849\r
30817668  X_100744436_C_T  rs183669380        X  100744436               C                T  -0.0132            NaN  ...                      NaN -0.0132          0.1626  0.96510         NaN       NaN      NaN  rs183669380\r
30817669   X_10031911_C_T  rs186607584        X   10031911               C                T  -0.1127            NaN  ...                      NaN -0.1127          0.0759  0.11550         NaN       NaN      NaN  rs186607584\r

[30817670 rows x 24 columns]
'''

#Drop hm_variant_id NA
#no hm_odds_ratio that is not NaN
#DROP hm_beta NA
#do not drop hm_odds_ratio due to ... all of them are NaN
chen_a1c.dropna(axis=0, subset=['hm_variant_id', 'hm_beta'], inplace=True)

'''
chen_a1c
            hm_variant_id      hm_rsid hm_chrom     hm_pos hm_other_allele hm_effect_allele  hm_beta  hm_odds_ratio  ...  effect_allele_frequency    beta  standard_error  p_value  odds_ratio  ci_lower ci_upper   variant_id\r
0          1_99534456_G_T   rs10875231        1   99534456               G                T   0.0011            NaN  ...                    0.251  0.0011          0.0015  0.70520         NaN       NaN      NaN   rs10875231\r
1           1_9939948_G_A  rs186077422        1    9939948               G                A   0.0136            NaN  ...                    0.008  0.0136          0.0322  0.71810         NaN       NaN      NaN  rs186077422\r
2          1_99534579_A_T  rs114947036        1   99534579               A                T   0.0160            NaN  ...                    0.999 -0.0160          0.0895  0.62260         NaN       NaN      NaN  rs114947036\r
3          1_99535129_T_C  rs140126637        1   99535129               T                C  -0.1146            NaN  ...                    0.999  0.1146          0.0807  0.08056         NaN       NaN      NaN  rs140126637\r
4          1_99535213_T_C  rs543738424        1   99535213               T                C   0.0719            NaN  ...                      NaN -0.0719          0.1649  0.73750         NaN       NaN      NaN  rs543738424\r
...                   ...          ...      ...        ...             ...              ...      ...            ...  ...                      ...     ...             ...      ...         ...       ...      ...            ...
30817665  X_100744278_G_A  rs140486085        X  100744278               G                A  -0.0027            NaN  ...                      NaN -0.0027          0.1321  0.93150         NaN       NaN      NaN  rs140486085\r
30817666  X_100744360_G_A    rs2021704        X  100744360               G                A   0.0018            NaN  ...                      NaN  0.0018          0.0018  0.17050         NaN       NaN      NaN    rs2021704\r
30817667  X_100744400_A_G  rs146574849        X  100744400               A                G  -0.0048            NaN  ...                      NaN  0.0048          0.0026  0.06514         NaN       NaN      NaN  rs146574849\r
30817668  X_100744436_C_T  rs183669380        X  100744436               C                T  -0.0132            NaN  ...                      NaN -0.0132          0.1626  0.96510         NaN       NaN      NaN  rs183669380\r
30817669   X_10031911_C_T  rs186607584        X   10031911               C                T  -0.1127            NaN  ...                      NaN -0.1127          0.0759  0.11550         NaN       NaN      NaN  rs186607584\r

[30655195 rows x 24 columns]
'''

#Drop invalid hm_codes
print(len(chen_a1c.index))
30655195
hm_code_mask = chen_a1c.hm_code.ge(14) | chen_a1c.hm_code.le(6)
hm_code_mask.sum()
4647639
chen_a1c.drop(chen_a1c[hm_code_mask].index, inplace=True)
print(len(chen_a1c.index))
26007556

'''
chen_a1c
            hm_variant_id      hm_rsid hm_chrom     hm_pos hm_other_allele hm_effect_allele  hm_beta  hm_odds_ratio  ...  effect_allele_frequency    beta  standard_error  p_value  odds_ratio  ci_lower ci_upper   variant_id\r
0          1_99534456_G_T   rs10875231        1   99534456               G                T   0.0011            NaN  ...                    0.251  0.0011          0.0015  0.70520         NaN       NaN      NaN   rs10875231\r
1           1_9939948_G_A  rs186077422        1    9939948               G                A   0.0136            NaN  ...                    0.008  0.0136          0.0322  0.71810         NaN       NaN      NaN  rs186077422\r
3          1_99535129_T_C  rs140126637        1   99535129               T                C  -0.1146            NaN  ...                    0.999  0.1146          0.0807  0.08056         NaN       NaN      NaN  rs140126637\r
4          1_99535213_T_C  rs543738424        1   99535213               T                C   0.0719            NaN  ...                      NaN -0.0719          0.1649  0.73750         NaN       NaN      NaN  rs543738424\r
5          1_99535271_C_T    rs6678176        1   99535271               C                T  -0.0003            NaN  ...                    0.314 -0.0003          0.0014  0.62140         NaN       NaN      NaN    rs6678176\r
...                   ...          ...      ...        ...             ...              ...      ...            ...  ...                      ...     ...             ...      ...         ...       ...      ...            ...
30817665  X_100744278_G_A  rs140486085        X  100744278               G                A  -0.0027            NaN  ...                      NaN -0.0027          0.1321  0.93150         NaN       NaN      NaN  rs140486085\r
30817666  X_100744360_G_A    rs2021704        X  100744360               G                A   0.0018            NaN  ...                      NaN  0.0018          0.0018  0.17050         NaN       NaN      NaN    rs2021704\r
30817667  X_100744400_A_G  rs146574849        X  100744400               A                G  -0.0048            NaN  ...                      NaN  0.0048          0.0026  0.06514         NaN       NaN      NaN  rs146574849\r
30817668  X_100744436_C_T  rs183669380        X  100744436               C                T  -0.0132            NaN  ...                      NaN -0.0132          0.1626  0.96510         NaN       NaN      NaN  rs183669380\r
30817669   X_10031911_C_T  rs186607584        X   10031911               C                T  -0.1127            NaN  ...                      NaN -0.1127          0.0759  0.11550         NaN       NaN      NaN  rs186607584\r

[26007556 rows x 24 columns]
'''
#need to type cast, etc.
remap_dict = {'X':23}
chen_a1c['hm_chrom'] = chen_a1c['hm_chrom'].replace(remap_dict)
chen_a1c['hm_chrom'] = chen_a1c['hm_chrom'].astype('int64')
chen_a1c['hm_pos'] = chen_a1c['hm_pos'].astype('int64')
chen_a1c = chen_a1c[chena1c_columns].sort_values(by=['hm_chrom', 'hm_pos'])

'''
chen_a1c
         hm_variant_id     hm_rsid  hm_chrom    hm_pos hm_other_allele hm_effect_allele   hm_beta  hm_odds_ratio  hm_ci_lower         hm_ci_upper  p_value
49967    1_1182895_C_T  rs61733845         1   1182895               C                T -0.011971         0.9881     0.914437  1.0676971763025762    0.762
49968    1_1185051_G_A   rs1320571         1   1185051               G                A  0.025668         1.0260     0.950742  1.1072146259925333    0.509
49969    1_1199862_A_C   rs9729550         1   1199862               A                C  0.001000         1.0010     0.953293  1.0510949308412958    0.968
49970    1_1205055_G_T   rs1815606         1   1205055               G                T  0.002996         1.0030     0.966544  1.0408306968055108    0.874
49971    1_1228424_C_T   rs7515488         1   1228424               C                T  0.019803         1.0200     0.975796   1.066206563525329    0.381
...                ...         ...       ...       ...             ...              ...       ...            ...          ...                 ...      ...
74524  22_50656498_C_T   rs9616810        22  50656498               C                T  0.006976         1.0070     0.968181  1.0473755842106767    0.728
74525  22_50667128_C_T   rs9616812        22  50667128               C                T -0.002002         0.9980     0.971055  1.0256922064673688    0.886
74526  22_50671564_T_C   rs9628185        22  50671564               T                C  0.001000         1.0010     0.968947  1.0341127907285765    0.952
74527  22_50695758_A_G   rs8135777        22  50695758               A                G  0.631804         1.8810     0.780813   4.531379050901968    0.159
74528  22_50718238_C_T   rs9628187        22  50718238               C                T -0.006018         0.9940     0.952710  1.0370799724102215    0.781

[132089 rows x 11 columns]
'''

chen_a1c.equals(chen_a1c.sort_values(by=['hm_chrom', 'hm_pos']))
oc_chen = ['hm_chrom', 'hm_pos','hm_other_allele', 'hm_effect_allele', 'hm_beta', 'p_value']

'''
chen_a1c[oc_chen].to_csv('chen_a1c_gwas_lz'+version+'.txt', sep='\t', index=False)
os.system("gzip -k " + 'chen_a1c_gwas_lz'+version+'.txt &')
'''
'''
for the uploads to locus zoom, the following parameters were selected:
    chr:hm_chrom
    pos:hm_pos
    ref allele: hm_other_allele
    alt allele: hm_effect_allele
    p-value: p_value
'''

'''
********************************************************************************
Compare effect direction across diseases

1. Merge each individual GWAS/IC data set around CLEC16A region
16:10846031-11346031 is the SLE metal defined region.
However, chr16:10,800,000-11,700,000 is chosen since several of the other
immune mediated diseases show increased signal in/near RMI2 and SOCS1 over and against
CLEC16A. X
hg38
16:10800000-11700000
hg19
chr16:10893857-11793856
2a. subset Beta and make heatmap with all markers passing GWS in any study X
2b. do this again for markers in common across all studies only X

3a. subset Z = Beta/SE and make heatmap with all markers passing GWS in any study
3b. do this again for markers in common across all studies only

4. make clustermap with whichever of the above is the best balance of clarity and comprehenison
to find which diseases/regions cluster together. X

5. meta-analysis on related genetic effects?
6. mendelian randomization?
********************************************************************************
'''

'''
MERGE THE CLEC16A AREA – 
'''
chr = 16
start = 10800000
end = 11700000

sawcer_ms
#gwas
cordell_pbc
#gwas
bentham_sle
ji_psc
#gwas
ferriera_atopy
demenais_asthma
onengut_t1d
#IC
sliz_ad
dubois_cd
liu_ibd
liu_cd
delange_ibd
delange_cd
tsoi_pso
#IC
hinks_jia
beecham_ms
#IC
liu_pbc
#IC
chen_a1c

'''
masks for clec16a region
'''
mask_sawcer_ms = sawcer_ms['hm_chrom'].eq(chr) & sawcer_ms['hm_pos'].ge(start) & sawcer_ms['hm_pos'].le(end)
mask_cordell_pbc = cordell_pbc['hm_chrom'].eq(chr) & cordell_pbc['hm_pos'].ge(start) & cordell_pbc['hm_pos'].le(end)
mask_bentham_sle = bentham_sle['hm_chrom'].eq(chr) & bentham_sle['hm_pos'].ge(start) & bentham_sle['hm_pos'].le(end)
mask_ji_psc = ji_psc['hm_chrom'].eq(chr) & ji_psc['hm_pos'].ge(start) & ji_psc['hm_pos'].le(end)
mask_ferriera_atopy = ferriera_atopy['hm_chrom'].eq(chr) & ferriera_atopy['hm_pos'].ge(start) & ferriera_atopy['hm_pos'].le(end)
mask_demenais_asthma = demenais_asthma['hm_chrom'].eq(chr) & demenais_asthma['hm_pos'].ge(start) & demenais_asthma['hm_pos'].le(end)
mask_onengut_t1d = onengut_t1d['hm_chrom'].eq(chr) & onengut_t1d['hm_pos'].ge(start) & onengut_t1d['hm_pos'].le(end)
mask_sliz_ad = sliz_ad['hm_chrom'].eq(chr) & sliz_ad['hm_pos'].ge(start) & sliz_ad['hm_pos'].le(end)
mask_dubois_cd = dubois_cd['hm_chrom'].eq(chr) & dubois_cd['hm_pos'].ge(start) & dubois_cd['hm_pos'].le(end)
mask_liu_ibd = liu_ibd['hm_chrom'].eq(chr) & liu_ibd['hm_pos'].ge(start) & liu_ibd['hm_pos'].le(end)
mask_liu_cd = liu_cd['hm_chrom'].eq(chr) & liu_cd['hm_pos'].ge(start) & liu_cd['hm_pos'].le(end)
mask_delange_ibd = delange_ibd['hm_chrom'].eq(chr) & delange_ibd['hm_pos'].ge(start) & delange_ibd['hm_pos'].le(end)
mask_delange_cd = delange_cd['hm_chrom'].eq(chr) & delange_cd['hm_pos'].ge(start) & delange_cd['hm_pos'].le(end)
mask_tsoi_pso = tsoi_pso['hm_chrom'].eq(chr) & tsoi_pso['hm_pos'].ge(start) & tsoi_pso['hm_pos'].le(end)
mask_hinks_jia = hinks_jia['hm_chrom'].eq(chr) & hinks_jia['hm_pos'].ge(start) & hinks_jia['hm_pos'].le(end)
mask_beecham_ms = beecham_ms['hm_chrom'].eq(chr) & beecham_ms['hm_pos'].ge(start) & beecham_ms['hm_pos'].le(end)
mask_liu_pbc = liu_pbc['hm_chrom'].eq(chr) & liu_pbc['hm_pos'].ge(start) & liu_pbc['hm_pos'].le(end)
mask_chen_a1c = chen_a1c['hm_chrom'].eq(chr) & chen_a1c['hm_pos'].ge(start) & chen_a1c['hm_pos'].le(end)

'''
slice the region with the masks
'''
sawcer_ms_CLEC16A = sawcer_ms[mask_sawcer_ms]
cordell_pbc_CLEC16A = cordell_pbc[mask_cordell_pbc]
bentham_sle_CLEC16A = bentham_sle[mask_bentham_sle]
ji_psc_CLEC16A = ji_psc[mask_ji_psc]
ferriera_atopy_CLEC16A = ferriera_atopy[mask_ferriera_atopy]
demenais_asthma_CLEC16A = demenais_asthma[mask_demenais_asthma]
onengut_t1d_CLEC16A = onengut_t1d[mask_onengut_t1d]
sliz_ad_CLEC16A = sliz_ad[mask_sliz_ad]
dubois_cd_CLEC16A = dubois_cd[mask_dubois_cd]
liu_ibd_CLEC16A = liu_ibd[mask_liu_ibd]
liu_cd_CLEC16A = liu_cd[mask_liu_cd]
delange_ibd_CLEC16A = delange_ibd[mask_delange_ibd]
delange_cd_CLEC16A = delange_cd[mask_delange_cd]
tsoi_pso_CLEC16A = tsoi_pso[mask_tsoi_pso]
hinks_jia_CLEC16A = hinks_jia[mask_hinks_jia]
beecham_ms_CLEC16A = beecham_ms[mask_beecham_ms]
liu_pbc_CLEC16A = liu_pbc[mask_liu_pbc]
chen_a1c_CLEC16A = chen_a1c[mask_chen_a1c]

sawcer_ms_CLEC16A = sawcer_ms_CLEC16A.assign(ZScore=lambda x: x['hm_beta']/x['standard_error'])
cordell_pbc_CLEC16A = cordell_pbc_CLEC16A.assign(ZScore=lambda x: x['hm_beta']/x['standard_error'])
bentham_sle_CLEC16A = bentham_sle_CLEC16A.assign(ZScore=lambda x: x['hm_beta']/x['standard_error'])
ji_psc_CLEC16A = ji_psc_CLEC16A.assign(ZScore=lambda x: x['hm_beta']/x['standard_error'])
ferriera_atopy_CLEC16A = ferriera_atopy_CLEC16A.assign(ZScore=lambda x: x['hm_beta']/x['standard_error'])
demenais_asthma_CLEC16A = demenais_asthma_CLEC16A.assign(ZScore=lambda x: x['hm_beta']/x['standard_error'])
onengut_t1d_CLEC16A = onengut_t1d_CLEC16A.assign(ZScore=lambda x: x['hm_beta']/x['standard_error'])
sliz_ad_CLEC16A = sliz_ad_CLEC16A.assign(ZScore=lambda x: x['hm_beta']/x['standard_error'])
dubois_cd_CLEC16A = dubois_cd_CLEC16A.assign(ZScore=lambda x: x['hm_beta']/x['standard_error'])
liu_ibd_CLEC16A = liu_ibd_CLEC16A.assign(ZScore=lambda x: x['hm_beta']/x['standard_error'])
liu_cd_CLEC16A = liu_cd_CLEC16A.assign(ZScore=lambda x: x['hm_beta']/x['standard_error'])
delange_ibd_CLEC16A = delange_ibd_CLEC16A.assign(ZScore=lambda x: x['hm_beta']/x['standard_error'])
delange_cd_CLEC16A = delange_cd_CLEC16A.assign(ZScore=lambda x: x['hm_beta']/x['standard_error'])
tsoi_pso_CLEC16A = tsoi_pso_CLEC16A.assign(ZScore=lambda x: x['hm_beta']/x['standard_error'])
hinks_jia_CLEC16A = hinks_jia_CLEC16A.assign(ZScore=lambda x: x['hm_beta']/x['standard_error'])
beecham_ms_CLEC16A = beecham_ms_CLEC16A.assign(ZScore=lambda x: x['hm_beta']/x['standard_error'])
liu_pbc_CLEC16A = liu_pbc_CLEC16A.assign(ZScore=lambda x: x['hm_beta']/x['standard_error'])
chen_a1c_CLEC16A = chen_a1c_CLEC16A.assign(ZScore=lambda x: x['hm_beta']/x['standard_error'])

sawcer_ms_CLEC16A = sawcer_ms_CLEC16A.assign(CP=lambda x: x['hm_chrom'].astype('str') + ":" + x['hm_pos'].astype('str'))
cordell_pbc_CLEC16A = cordell_pbc_CLEC16A.assign(CP=lambda x: x['hm_chrom'].astype('str') + ":" + x['hm_pos'].astype('str'))
bentham_sle_CLEC16A = bentham_sle_CLEC16A.assign(CP=lambda x: x['hm_chrom'].astype('str') + ":" + x['hm_pos'].astype('str'))
ji_psc_CLEC16A = ji_psc_CLEC16A.assign(CP=lambda x: x['hm_chrom'].astype('str') + ":" + x['hm_pos'].astype('str'))
ferriera_atopy_CLEC16A = ferriera_atopy_CLEC16A.assign(CP=lambda x: x['hm_chrom'].astype('str') + ":" + x['hm_pos'].astype('str'))
demenais_asthma_CLEC16A = demenais_asthma_CLEC16A.assign(CP=lambda x: x['hm_chrom'].astype('str') + ":" + x['hm_pos'].astype('str'))
onengut_t1d_CLEC16A = onengut_t1d_CLEC16A.assign(CP=lambda x: x['hm_chrom'].astype('str') + ":" + x['hm_pos'].astype('str'))
sliz_ad_CLEC16A = sliz_ad_CLEC16A.assign(CP=lambda x: x['hm_chrom'].astype('str') + ":" + x['hm_pos'].astype('str'))
dubois_cd_CLEC16A = dubois_cd_CLEC16A.assign(CP=lambda x: x['hm_chrom'].astype('str') + ":" + x['hm_pos'].astype('str'))
liu_ibd_CLEC16A = liu_ibd_CLEC16A.assign(CP=lambda x: x['hm_chrom'].astype('str') + ":" + x['hm_pos'].astype('str'))
liu_cd_CLEC16A = liu_cd_CLEC16A.assign(CP=lambda x: x['hm_chrom'].astype('str') + ":" + x['hm_pos'].astype('str'))
delange_ibd_CLEC16A = delange_ibd_CLEC16A.assign(CP=lambda x: x['hm_chrom'].astype('str') + ":" + x['hm_pos'].astype('str'))
delange_cd_CLEC16A = delange_cd_CLEC16A.assign(CP=lambda x: x['hm_chrom'].astype('str') + ":" + x['hm_pos'].astype('str'))
tsoi_pso_CLEC16A = tsoi_pso_CLEC16A.assign(CP=lambda x: x['hm_chrom'].astype('str') + ":" + x['hm_pos'].astype('str'))
hinks_jia_CLEC16A = hinks_jia_CLEC16A.assign(CP=lambda x: x['hm_chrom'].astype('str') + ":" + x['hm_pos'].astype('str'))
beecham_ms_CLEC16A = beecham_ms_CLEC16A.assign(CP=lambda x: x['hm_chrom'].astype('str') + ":" + x['hm_pos'].astype('str'))
liu_pbc_CLEC16A = liu_pbc_CLEC16A.assign(CP=lambda x: x['hm_chrom'].astype('str') + ":" + x['hm_pos'].astype('str'))
chen_a1c_CLEC16A = chen_a1c_CLEC16A.assign(CP=lambda x: x['hm_chrom'].astype('str') + ":" + x['hm_pos'].astype('str'))

sawcer_ms_CLEC16A.drop_duplicates(subset=['CP'], keep='first', inplace=True)
cordell_pbc_CLEC16A.drop_duplicates(subset=['CP'], keep='first', inplace=True)
bentham_sle_CLEC16A.drop_duplicates(subset=['CP'], keep='first', inplace=True)
ji_psc_CLEC16A.drop_duplicates(subset=['CP'], keep='first', inplace=True)
ferriera_atopy_CLEC16A.drop_duplicates(subset=['CP'], keep='first', inplace=True)
demenais_asthma_CLEC16A.drop_duplicates(subset=['CP'], keep='first', inplace=True)
onengut_t1d_CLEC16A.drop_duplicates(subset=['CP'], keep='first', inplace=True)
sliz_ad_CLEC16A.drop_duplicates(subset=['CP'], keep='first', inplace=True)
dubois_cd_CLEC16A.drop_duplicates(subset=['CP'], keep='first', inplace=True)
liu_ibd_CLEC16A.drop_duplicates(subset=['CP'], keep='first', inplace=True)
liu_cd_CLEC16A.drop_duplicates(subset=['CP'], keep='first', inplace=True)
delange_ibd_CLEC16A.drop_duplicates(subset=['CP'], keep='first', inplace=True)
delange_cd_CLEC16A.drop_duplicates(subset=['CP'], keep='first', inplace=True)
tsoi_pso_CLEC16A.drop_duplicates(subset=['CP'], keep='first', inplace=True)
hinks_jia_CLEC16A.drop_duplicates(subset=['CP'], keep='first', inplace=True)
beecham_ms_CLEC16A.drop_duplicates(subset=['CP'], keep='first', inplace=True)
liu_pbc_CLEC16A.drop_duplicates(subset=['CP'], keep='first', inplace=True)
chen_a1c_CLEC16A.drop_duplicates(subset=['CP'], keep='first', inplace=True)

study_codes = ["s_ms","c_pbc","b_sle","j_psc","f_all","d_asm","o_t1d","s_ad","d_ced","l_ibd","l_cd","d_ibd","d_cd","t_pso","h_jia","b_ms","l_pbc","c_a1c"]
len(study_codes)
'''
assign unambigouous column names to facilitate merging later
force the 'CP' column to not be suffixed
'''
#Merge based on hm_variant_id
sawcer_ms_CLEC16A.columns = ['hm_variant_id']+[str(x) + "_" + study_codes[0] for x in sawcer_ms_CLEC16A.columns.to_list()[1:len(sawcer_ms_CLEC16A.columns.to_list())-1]]+['CP']
cordell_pbc_CLEC16A.columns = ['hm_variant_id']+[str(x) + "_" + study_codes[1] for x in cordell_pbc_CLEC16A.columns.to_list()[1:len(cordell_pbc_CLEC16A.columns.to_list())-1]]+['CP']
bentham_sle_CLEC16A.columns = ['hm_variant_id']+[str(x) + "_" + study_codes[2] for x in bentham_sle_CLEC16A.columns.to_list()[1:len(bentham_sle_CLEC16A.columns.to_list())-1]]+['CP']
ji_psc_CLEC16A.columns = ['hm_variant_id']+[str(x) + "_" + study_codes[4] for x in ji_psc_CLEC16A.columns.to_list()[1:len(ji_psc_CLEC16A.columns.to_list())-1]]+['CP']
ferriera_atopy_CLEC16A.columns = ['hm_variant_id']+[str(x) + "_" + study_codes[5] for x in ferriera_atopy_CLEC16A.columns.to_list()[1:len(ferriera_atopy_CLEC16A.columns.to_list())-1]]+['CP']
demenais_asthma_CLEC16A.columns = ['hm_variant_id']+[str(x) + "_" + study_codes[6] for x in demenais_asthma_CLEC16A.columns.to_list()[1:len(demenais_asthma_CLEC16A.columns.to_list())-1]]+['CP']
onengut_t1d_CLEC16A.columns = ['hm_variant_id']+[str(x) + "_" + study_codes[7] for x in onengut_t1d_CLEC16A.columns.to_list()[1:len(onengut_t1d_CLEC16A.columns.to_list())-1]]+['CP']
sliz_ad_CLEC16A.columns = ['hm_variant_id']+[str(x) + "_" + study_codes[8] for x in sliz_ad_CLEC16A.columns.to_list()[1:len(sliz_ad_CLEC16A.columns.to_list())-1]]+['CP']
dubois_cd_CLEC16A.columns = ['hm_variant_id']+[str(x) + "_" + study_codes[9] for x in dubois_cd_CLEC16A.columns.to_list()[1:len(dubois_cd_CLEC16A.columns.to_list())-1]]+['CP']
liu_ibd_CLEC16A.columns = ['hm_variant_id']+[str(x) + "_" + study_codes[10] for x in liu_ibd_CLEC16A.columns.to_list()[1:len(liu_ibd_CLEC16A.columns.to_list())-1]]+['CP']
liu_cd_CLEC16A.columns = ['hm_variant_id']+[str(x) + "_" + study_codes[11] for x in liu_cd_CLEC16A.columns.to_list()[1:len(liu_cd_CLEC16A.columns.to_list())-1]]+['CP']
delange_ibd_CLEC16A.columns = ['hm_variant_id']+[str(x) + "_" + study_codes[12] for x in delange_ibd_CLEC16A.columns.to_list()[1:len(delange_ibd_CLEC16A.columns.to_list())-1]]+['CP']
delange_cd_CLEC16A.columns = ['hm_variant_id']+[str(x) + "_" + study_codes[13] for x in delange_cd_CLEC16A.columns.to_list()[1:len(delange_cd_CLEC16A.columns.to_list())-1]]+['CP']
tsoi_pso_CLEC16A.columns = ['hm_variant_id']+[str(x) + "_" + study_codes[14] for x in tsoi_pso_CLEC16A.columns.to_list()[1:len(tsoi_pso_CLEC16A.columns.to_list())-1]]+['CP']
hinks_jia_CLEC16A.columns = ['hm_variant_id']+[str(x) + "_" + study_codes[15] for x in hinks_jia_CLEC16A.columns.to_list()[1:len(hinks_jia_CLEC16A.columns.to_list())-1]]+['CP']
beecham_ms_CLEC16A.columns = ['hm_variant_id']+[str(x) + "_" + study_codes[16] for x in beecham_ms_CLEC16A.columns.to_list()[1:len(beecham_ms_CLEC16A.columns.to_list())-1]]+['CP']
liu_pbc_CLEC16A.columns = ['hm_variant_id']+[str(x) + "_" + study_codes[17] for x in liu_pbc_CLEC16A.columns.to_list()[1:len(liu_pbc_CLEC16A.columns.to_list())-1]]+['CP']
chen_a1c_CLEC16A.columns = ['hm_variant_id']+[str(x) + "_" + study_codes[18] for x in chen_a1c_CLEC16A.columns.to_list()[1:len(chen_a1c_CLEC16A.columns.to_list())-1]]+['CP']
'''
for merging based on CP only
sawcer_ms_CLEC16A.columns = [str(x) + "_" + study_codes[0] for x in sawcer_ms_CLEC16A.columns.to_list()[0:len(sawcer_ms_CLEC16A.columns.to_list())-1]]+['CP']
cordell_pbc_CLEC16A.columns = [str(x) + "_" + study_codes[1] for x in cordell_pbc_CLEC16A.columns.to_list()[0:len(cordell_pbc_CLEC16A.columns.to_list())-1]]+['CP']
bentham_sle_CLEC16A.columns = [str(x) + "_" + study_codes[2] for x in bentham_sle_CLEC16A.columns.to_list()[0:len(bentham_sle_CLEC16A.columns.to_list())-1]]+['CP']
ji_psc_CLEC16A.columns = [str(x) + "_" + study_codes[4] for x in ji_psc_CLEC16A.columns.to_list()[0:len(ji_psc_CLEC16A.columns.to_list())-1]]+['CP']
ferriera_atopy_CLEC16A.columns = [str(x) + "_" + study_codes[5] for x in ferriera_atopy_CLEC16A.columns.to_list()[0:len(ferriera_atopy_CLEC16A.columns.to_list())-1]]+['CP']
demenais_asthma_CLEC16A.columns = [str(x) + "_" + study_codes[6] for x in demenais_asthma_CLEC16A.columns.to_list()[0:len(demenais_asthma_CLEC16A.columns.to_list())-1]]+['CP']
onengut_t1d_CLEC16A.columns = [str(x) + "_" + study_codes[7] for x in onengut_t1d_CLEC16A.columns.to_list()[0:len(onengut_t1d_CLEC16A.columns.to_list())-1]]+['CP']
sliz_ad_CLEC16A.columns = [str(x) + "_" + study_codes[8] for x in sliz_ad_CLEC16A.columns.to_list()[0:len(sliz_ad_CLEC16A.columns.to_list())-1]]+['CP']
dubois_cd_CLEC16A.columns = [str(x) + "_" + study_codes[9] for x in dubois_cd_CLEC16A.columns.to_list()[0:len(dubois_cd_CLEC16A.columns.to_list())-1]]+['CP']
liu_ibd_CLEC16A.columns = [str(x) + "_" + study_codes[10] for x in liu_ibd_CLEC16A.columns.to_list()[0:len(liu_ibd_CLEC16A.columns.to_list())-1]]+['CP']
liu_cd_CLEC16A.columns = [str(x) + "_" + study_codes[11] for x in liu_cd_CLEC16A.columns.to_list()[0:len(liu_cd_CLEC16A.columns.to_list())-1]]+['CP']
delange_ibd_CLEC16A.columns = [str(x) + "_" + study_codes[12] for x in delange_ibd_CLEC16A.columns.to_list()[0:len(delange_ibd_CLEC16A.columns.to_list())-1]]+['CP']
delange_cd_CLEC16A.columns = [str(x) + "_" + study_codes[13] for x in delange_cd_CLEC16A.columns.to_list()[0:len(delange_cd_CLEC16A.columns.to_list())-1]]+['CP']
tsoi_pso_CLEC16A.columns = [str(x) + "_" + study_codes[14] for x in tsoi_pso_CLEC16A.columns.to_list()[0:len(tsoi_pso_CLEC16A.columns.to_list())-1]]+['CP']
hinks_jia_CLEC16A.columns = [str(x) + "_" + study_codes[15] for x in hinks_jia_CLEC16A.columns.to_list()[0:len(hinks_jia_CLEC16A.columns.to_list())-1]]+['CP']
beecham_ms_CLEC16A.columns = [str(x) + "_" + study_codes[16] for x in beecham_ms_CLEC16A.columns.to_list()[0:len(beecham_ms_CLEC16A.columns.to_list())-1]]+['CP']
liu_pbc_CLEC16A.columns = [str(x) + "_" + study_codes[17] for x in liu_pbc_CLEC16A.columns.to_list()[0:len(liu_pbc_CLEC16A.columns.to_list())-1]]+['CP']
chen_a1c_CLEC16A.columns = [str(x) + "_" + study_codes[18] for x in chen_a1c_CLEC16A.columns.to_list()[0:len(chen_a1c_CLEC16A.columns.to_list())-1]]+['CP']
'''
sawcer_ms_CLEC16A
cordell_pbc_CLEC16A
bentham_sle_CLEC16A
ji_psc_CLEC16A
ferriera_atopy_CLEC16A
demenais_asthma_CLEC16A
onengut_t1d_CLEC16A
sliz_ad_CLEC16A
dubois_cd_CLEC16A
liu_ibd_CLEC16A
liu_cd_CLEC16A
delange_ibd_CLEC16A
delange_cd_CLEC16A
tsoi_pso_CLEC16A
hinks_jia_CLEC16A
beecham_ms_CLEC16A
liu_pbc_CLEC16A
chen_a1c_CLEC16A

#Merge on hm_variant_id
studies = [sawcer_ms_CLEC16A, cordell_pbc_CLEC16A, bentham_sle_CLEC16A,ji_psc_CLEC16A, ferriera_atopy_CLEC16A, demenais_asthma_CLEC16A, onengut_t1d_CLEC16A, sliz_ad_CLEC16A, dubois_cd_CLEC16A, liu_ibd_CLEC16A, liu_cd_CLEC16A, delange_ibd_CLEC16A, delange_cd_CLEC16A, tsoi_pso_CLEC16A, hinks_jia_CLEC16A, beecham_ms_CLEC16A, liu_pbc_CLEC16A, chen_a1c_CLEC16A]
all_marker_table = pd.DataFrame([])
common_marker_table = pd.DataFrame([])
#study_names = ["sawcer_ms","cordell_pbc","bentham_sle","ji_psc","ferriera_atopy","demenais_asthma","onengut_t1d","sliz_ad","dubois_cd","liu_ibd","liu_cd","delange_ibd","delange_cd","tsoi_pso","hinks_jia","beecham_ms","liu_pbc","chen_a1c"]
all_marker_table = pd.merge(studies[0], studies[1], on='hm_variant_id', how='left')
common_marker_table = pd.merge(studies[0], studies[1], on='hm_variant_id', how='inner')
#common_marker_table = pd.merge(common_marker_table, studies[2], on='CP', how='left')
i=0
for study in studies[2:len(studies)]:
    print(study_codes[i])
    all_marker_table = pd.merge(all_marker_table, study, on='hm_variant_id', how='left')
    common_marker_table = pd.merge(common_marker_table, study, on='hm_variant_id', how='outer')
    print(i)
    i=i+1


'''
Merge on CP Only
studies = [sawcer_ms_CLEC16A, cordell_pbc_CLEC16A, bentham_sle_CLEC16A, ji_psc_CLEC16A, ferriera_atopy_CLEC16A, demenais_asthma_CLEC16A, onengut_t1d_CLEC16A, sliz_ad_CLEC16A, dubois_cd_CLEC16A, liu_ibd_CLEC16A, liu_cd_CLEC16A, delange_ibd_CLEC16A, delange_cd_CLEC16A, tsoi_pso_CLEC16A, hinks_jia_CLEC16A, beecham_ms_CLEC16A, liu_pbc_CLEC16A, chen_a1c_CLEC16A]
common_marker_table = pd.DataFrame([])
#study_names = ["sawcer_ms","cordell_pbc","bentham_sle","ji_psc","ferriera_atopy","demenais_asthma","onengut_t1d","sliz_ad","dubois_cd","liu_ibd","liu_cd","delange_ibd","delange_cd","tsoi_pso","hinks_jia","beecham_ms","liu_pbc","chen_a1c"]
common_marker_table = pd.merge(studies[0], studies[1], on='CP', how='left')
#common_marker_table = pd.merge(common_marker_table, studies[2], on='CP', how='left')
i=0
for study in studies[2:len(studies)]:
    print(study_codes[i])
    common_marker_table = pd.merge(common_marker_table, study, on='CP', how='left')
    print(i)
    i=i+1

'''
#common_marker_table = all_marker_table
cmt_columns = common_marker_table.columns.to_list()
filter_beta = filter(lambda a: 'hm_beta' in a, cmt_columns)
beta_columns = list(filter_beta)

filter_se = filter(lambda a: 'standard_error' in a, cmt_columns)
se_columns = list(filter_se)

filter_Z = filter(lambda a:'ZScore' in a, cmt_columns)
z_columns = list(filter_Z)
#version = '2022-10-01v2-'
#version = '2022-10-12-'
'''
********************************************************************************
Plot: Betas ClusterMap – ALL bentham markers, ward clustering, cluster rows & columns
********************************************************************************
'''
betas_slice = all_marker_table[['CP']+beta_columns].copy().sort_values(by=['CP'],ascending=False)
betas_slice = betas_slice.fillna(0)
betas_slice = betas_slice.set_index('CP')

plt.clf()
sns.set_theme()
colormap="vlag"
dataplot2 = sns.clustermap(betas_slice, center=0, cmap=colormap, col_cluster=True, row_cluster=True,
                    dendrogram_ratio=(.1, .3),
                    cbar_pos=(.05, .7, .05, .2),
                    linewidths=.75, figsize=(13, 26),yticklabels=True, method='ward')

plt.tight_layout()
colorscheme = 'seaborn_'+colormap
analysisscheme ='betas_only_'+"all_makers_ward_row_col_cluster"
plt.savefig(version+analysisscheme+colorscheme+'.png')

'''
********************************************************************************
Plot: Betas ClusterMap – ALL bentham markers, ward clustering, cluster columns only
********************************************************************************
'''
betas_slice = all_marker_table[['CP']+beta_columns].copy().sort_values(by=['CP'],ascending=False)
betas_slice = betas_slice.fillna(0)
betas_slice = betas_slice.set_index('CP')

plt.clf()
sns.set_theme()
colormap="vlag"
dataplot2 = sns.clustermap(betas_slice, center=0, cmap=colormap, col_cluster=True, row_cluster=False,
                    dendrogram_ratio=(.1, .3),
                    cbar_pos=(.05, .7, .05, .2),
                    linewidths=.75, figsize=(13, 26),yticklabels=True, method='ward')

plt.tight_layout()
colorscheme = 'seaborn_'+colormap
analysisscheme ='betas_only_'+"all_makers_ward_only_col_cluster"
plt.savefig(version+analysisscheme+colorscheme+'.png')


'''
subset GWS only
'''

cmt_columns = all_marker_table.columns.to_list()
filter_p_value = filter(lambda a: 'p_value' in a, cmt_columns)
p_value_columns = list(filter_p_value)
gws_mask = all_marker_table[p_value_columns[0]].le(5E-8)
for p_value_column in p_value_columns[1:len(p_value_columns)]:
    gws_mask = gws_mask | all_marker_table[p_value_column].le(5E-8)

betas_slice = all_marker_table[['CP', 'hm_rsid_b_sle']+beta_columns].copy()[gws_mask]
betas_slice = all_marker_table[['CP']+beta_columns].copy()[gws_mask].sort_values(by=['CP'],ascending=False)
betas_slice = betas_slice.fillna(0)
betas_slice = betas_slice.set_index('CP')

'''
********************************************************************************
Plot: Betas ClusterMap – GWS markers, ward clustering, cluster columns only
********************************************************************************
'''
plt.clf()
sns.set_theme()
colormap="vlag"
dataplot2 = sns.clustermap(betas_slice, center=0, cmap=colormap, col_cluster=True, row_cluster=False,
                    dendrogram_ratio=(.1, .3),
                    cbar_pos=(.05, .7, .05, .2),
                    linewidths=.75, figsize=(13, 26),yticklabels=True, method='ward')

plt.tight_layout()
colorscheme = 'seaborn_'+colormap
analysisscheme ='betas_only_'+"gws_makers_ward_only_col_cluster"
plt.savefig(version+analysisscheme+colorscheme+'.png')

'''
********************************************************************************
Plot: Betas ClusterMap – GWS markers, ward clustering, cluster rows & columns
********************************************************************************
'''

plt.clf()
sns.set_theme()
colormap="vlag"
dataplot2 = sns.clustermap(betas_slice, center=0, cmap=colormap, col_cluster=True, row_cluster=True,
                    dendrogram_ratio=(.1, .3),
                    cbar_pos=(.05, .7, .05, .2),
                    linewidths=.75, figsize=(13, 26),yticklabels=True, method='ward')

plt.tight_layout()
colorscheme = 'seaborn_'+colormap
analysisscheme ='betas_only_'+"gws_makers_ward_row_col_cluster"
plt.savefig(version+analysisscheme+colorscheme+'.png')

'''
subset GWS & common only
'''

cmt_columns = common_marker_table.columns.to_list()
filter_p_value = filter(lambda a: 'p_value' in a, cmt_columns)
p_value_columns = list(filter_p_value)
cgws_mask = common_marker_table[p_value_columns[0]].le(5E-8)
for p_value_column in p_value_columns[1:len(p_value_columns)]:
    cgws_mask = cgws_mask | common_marker_table[p_value_column].le(5E-8)

common_marker_table[cgws_mask].to_csv('CommonGWS_markers.txt', sep='\t')
betas_slice = common_marker_table[['CP']+beta_columns].copy()[cgws_mask]
betas_slice = betas_slice.dropna(axis=0)
betas_slice = betas_slice.set_index('CP')
betas_slice = betas_slice.sort_values(by=['CP'],ascending=False)

'''
Almost – need to go back and merge on hm_variant_id
some cp combos hitting multiple rsids
'''
#betas_slice.to_csv('common_markers_main_fig.txt', sep='\t')

'''
********************************************************************************
Plot: Betas ClusterMap – Common GWS markers, ward clustering, cluster columns only
********************************************************************************
'''
plt.clf()
sns.set_theme()
colormap="vlag"
dataplot2 = sns.clustermap(betas_slice, center=0, cmap=colormap, col_cluster=True, row_cluster=False,
                    dendrogram_ratio=(.1, .3),
                    cbar_pos=(.05, .7, .05, .2),
                    linewidths=.75, figsize=(13, 26),yticklabels=True, method='ward')

plt.tight_layout()
colorscheme = 'seaborn_'+colormap
analysisscheme ='betas_only_'+"commonGWS_makers_ward_only_col_cluster"
plt.savefig(version+analysisscheme+colorscheme+'.png')
plt.savefig(version+analysisscheme+colorscheme+'.svg')

betas_slice.to_csv('MainFigureCoordinates.txt', sep='\t')

'''
********************************************************************************
Plot: Betas ClusterMap – Common GWS markers, ward clustering, cluster rows & columns
********************************************************************************
'''

plt.clf()
sns.set_theme()
colormap="vlag"
dataplot2 = sns.clustermap(betas_slice, center=0, cmap=colormap, col_cluster=True, row_cluster=True,
                    dendrogram_ratio=(.1, .3),
                    cbar_pos=(.05, .7, .05, .2),
                    linewidths=.75, figsize=(13, 26),yticklabels=True, method='ward')

plt.tight_layout()
colorscheme = 'seaborn_'+colormap
analysisscheme ='betas_only_'+"commonGWS_makers_ward_row_col_cluster"
plt.savefig(version+analysisscheme+colorscheme+'.png')

'''
subset common only
'''

betas_slice = common_marker_table[['CP']+beta_columns].copy().sort_values(by=['CP'],ascending=False)
betas_slice = betas_slice.dropna(axis=0)
betas_slice = betas_slice.set_index('CP')

'''
********************************************************************************
Plot: Betas ClusterMap – Common markers, ward clustering, cluster columns only
********************************************************************************
'''
plt.clf()
sns.set_theme()
colormap="vlag"
dataplot2 = sns.clustermap(betas_slice, center=0, cmap=colormap, col_cluster=True, row_cluster=False,
                    dendrogram_ratio=(.1, .3),
                    cbar_pos=(.05, .7, .05, .2),
                    linewidths=.75, figsize=(13, 26),yticklabels=True, method='ward')

plt.tight_layout()
colorscheme = 'seaborn_'+colormap
analysisscheme ='betas_only_'+"common_makers_ward_only_col_cluster"
plt.savefig(version+analysisscheme+colorscheme+'.png')

'''
********************************************************************************
Plot: Betas ClusterMap – Common markers, ward clustering, cluster rows & columns
********************************************************************************
'''

plt.clf()
sns.set_theme()
colormap="vlag"
dataplot2 = sns.clustermap(betas_slice, center=0, cmap=colormap, col_cluster=True, row_cluster=True,
                    dendrogram_ratio=(.1, .3),
                    cbar_pos=(.05, .7, .05, .2),
                    linewidths=.75, figsize=(13, 26),yticklabels=True, method='ward')

plt.tight_layout()
colorscheme = 'seaborn_'+colormap
analysisscheme ='betas_only_'+"common_makers_ward_row_col_cluster"
plt.savefig(version+analysisscheme+colorscheme+'.png')


'''
********************************************************************************
********************************************************************************
                                *********************************************
                            *********************************************
                        *** Z SCORES ***
                    *********************************************
                *********************************************
********************************************************************************
********************************************************************************
'''

'''
********************************************************************************
Plot: Z score ClusterMap – ALL markers, ward clustering, cluster rows & columns
********************************************************************************
'''

zs_slice = all_marker_table[['CP']+z_columns].copy().sort_values(by=['CP'],ascending=False)
zs_slice = zs_slice.fillna(0)
zs_slice = zs_slice.set_index('CP')

plt.clf()
sns.set_theme()
colormap="vlag"
dataplot2 = sns.clustermap(zs_slice, center=0, cmap=colormap, col_cluster=True, row_cluster=True,
                    dendrogram_ratio=(.1, .3),
                    cbar_pos=(.05, .7, .05, .2),
                    linewidths=.75, figsize=(13, 26),yticklabels=True, method='ward')

plt.tight_layout()
colorscheme = 'seaborn_'+colormap
analysisscheme ='Zs_only_'+"all_makers_ward_row_col_cluster"
plt.savefig(version+analysisscheme+colorscheme+'.png')

'''
********************************************************************************
Plot: Z Score ClusterMap – ALL markers, ward clustering, cluster columns only
********************************************************************************
'''
zs_slice = all_marker_table[['CP']+z_columns].copy().sort_values(by=['CP'],ascending=False)
zs_slice = zs_slice.fillna(0)
zs_slice = zs_slice.set_index('CP')

plt.clf()
sns.set_theme()
colormap="vlag"
dataplot2 = sns.clustermap(zs_slice, center=0, cmap=colormap, col_cluster=True, row_cluster=False,
                    dendrogram_ratio=(.1, .3),
                    cbar_pos=(.05, .7, .05, .2),
                    linewidths=.75, figsize=(13, 26),yticklabels=True, method='ward')

plt.tight_layout()
colorscheme = 'seaborn_'+colormap
analysisscheme ='Zs_only_'+"all_makers_ward_only_col_cluster"
plt.savefig(version+analysisscheme+colorscheme+'.png')


'''
subset GWS only
'''

cmt_columns = all_marker_table.columns.to_list()
filter_p_value = filter(lambda a: 'p_value' in a, cmt_columns)
p_value_columns = list(filter_p_value)
gws_mask = all_marker_table[p_value_columns[0]].le(5E-8)
for p_value_column in p_value_columns[1:len(p_value_columns)]:
    gws_mask = gws_mask | common_marker_table[p_value_column].le(5E-8)

zs_slice = all_marker_table[['CP']+z_columns].copy()[gws_mask].sort_values(by=['CP'],ascending=False)
zs_slice = zs_slice.fillna(0)
zs_slice = zs_slice.set_index('CP')

'''
********************************************************************************
Plot: Z Score ClusterMap – GWS markers, ward clustering, cluster columns only
********************************************************************************
'''
plt.clf()
sns.set_theme()
colormap="vlag"
dataplot2 = sns.clustermap(zs_slice, center=0, cmap=colormap, col_cluster=True, row_cluster=False,
                    dendrogram_ratio=(.1, .3),
                    cbar_pos=(.05, .7, .05, .2),
                    linewidths=.75, figsize=(13, 26),yticklabels=True, method='ward')

plt.tight_layout()
colorscheme = 'seaborn_'+colormap
analysisscheme ='Zs_only_'+"gws_makers_ward_only_col_cluster"
plt.savefig(version+analysisscheme+colorscheme+'.png')

'''
********************************************************************************
Plot: Z Score ClusterMap – GWS markers, ward clustering, cluster rows & columns
********************************************************************************
'''

plt.clf()
sns.set_theme()
colormap="vlag"
dataplot2 = sns.clustermap(zs_slice, center=0, cmap=colormap, col_cluster=True, row_cluster=True,
                    dendrogram_ratio=(.1, .3),
                    cbar_pos=(.05, .7, .05, .2),
                    linewidths=.75, figsize=(13, 26),yticklabels=True, method='ward')

plt.tight_layout()
colorscheme = 'seaborn_'+colormap
analysisscheme ='Zs_only_'+"gws_makers_ward_row_col_cluster"
plt.savefig(version+analysisscheme+colorscheme+'.png')

'''
subset GWS & common only
'''
cmt_columns = common_marker_table.columns.to_list()
filter_p_value = filter(lambda a: 'p_value' in a, cmt_columns)
p_value_columns = list(filter_p_value)
cgws_mask = common_marker_table[p_value_columns[0]].le(5E-8)
for p_value_column in p_value_columns[1:len(p_value_columns)]:
    cgws_mask = cgws_mask | common_marker_table[p_value_column].le(5E-8)

zs_slice = common_marker_table[['CP']+z_columns].copy()[cgws_mask].sort_values(by=['CP'],ascending=False)
zs_slice = zs_slice.fillna(0)
zs_slice = zs_slice.set_index('CP')


'''
********************************************************************************
Plot: Z Score ClusterMap – Common GWS markers, ward clustering, cluster columns only
********************************************************************************
'''
plt.clf()
sns.set_theme()
colormap="vlag"
dataplot2 = sns.clustermap(zs_slice, center=0, cmap=colormap, col_cluster=True, row_cluster=False,
                    dendrogram_ratio=(.1, .3),
                    cbar_pos=(.05, .7, .05, .2),
                    linewidths=.75, figsize=(13, 26),yticklabels=True, method='ward')

plt.tight_layout()
colorscheme = 'seaborn_'+colormap
analysisscheme ='Zs_only_'+"commonGWS_makers_ward_only_col_cluster"
plt.savefig(version+analysisscheme+colorscheme+'.png')

'''
********************************************************************************
Plot: Z Score ClusterMap – Common GWS markers, ward clustering, cluster rows & columns
********************************************************************************
'''

plt.clf()
sns.set_theme()
colormap="vlag"
dataplot2 = sns.clustermap(zs_slice, center=0, cmap=colormap, col_cluster=True, row_cluster=True,
                    dendrogram_ratio=(.1, .3),
                    cbar_pos=(.05, .7, .05, .2),
                    linewidths=.75, figsize=(13, 26),yticklabels=True, method='ward')

plt.tight_layout()
colorscheme = 'seaborn_'+colormap
analysisscheme ='Zs_only_'+"commonGWS_makers_ward_row_col_cluster"
plt.savefig(version+analysisscheme+colorscheme+'.png')

'''
subset common only
'''

zs_slice = common_marker_table[['CP']+z_columns].copy().sort_values(by=['CP'],ascending=False)
zs_slice = zs_slice.dropna(axis=0)
zs_slice = zs_slice.set_index('CP')

'''
********************************************************************************
Plot: Z Score ClusterMap – Common markers, ward clustering, cluster columns only
********************************************************************************
'''
plt.clf()
sns.set_theme()
colormap="vlag"
dataplot2 = sns.clustermap(zs_slice, center=0, cmap=colormap, col_cluster=True, row_cluster=False,
                    dendrogram_ratio=(.1, .3),
                    cbar_pos=(.05, .7, .05, .2),
                    linewidths=.75, figsize=(13, 26),yticklabels=True, method='ward')

plt.tight_layout()
colorscheme = 'seaborn_'+colormap
analysisscheme ='Zs_only_'+"common_makers_ward_only_col_cluster"
plt.savefig(version+analysisscheme+colorscheme+'.png')

'''
********************************************************************************
Plot: Z Score  ClusterMap – Common markers, ward clustering, cluster rows & columns
********************************************************************************
'''

plt.clf()
sns.set_theme()
colormap="vlag"
dataplot2 = sns.clustermap(zs_slice, center=0, cmap=colormap, col_cluster=True, row_cluster=True,
                    dendrogram_ratio=(.1, .3),
                    cbar_pos=(.05, .7, .05, .2),
                    linewidths=.75, figsize=(13, 26),yticklabels=True, method='ward')

plt.tight_layout()
colorscheme = 'seaborn_'+colormap
analysisscheme ='Zs_only'+"common_makers_ward_row_col_cluster"
plt.savefig(version+analysisscheme+colorscheme+'.png')

common_marker_table.to_csv(version+'CLEC16A_GWAS_Summary_cmt.txt', sep='\t', index=True)
all_marker_table.to_csv(version+'CLEC16A_GWAS_Summary_amt.txt', sep='\t', index=True)

#points = pd.read_csv('MainFigureCoordinates.txt', sep='\t')

'''
slice betas for common and gws markers only
'''
cmt_columns = common_marker_table.columns.to_list()
filter_p_value = filter(lambda a: 'p_value' in a, cmt_columns)
p_value_columns = list(filter_p_value)
cgws_mask = common_marker_table[p_value_columns[0]].le(5E-8)
for p_value_column in p_value_columns[1:len(p_value_columns)]:
    cgws_mask = cgws_mask | common_marker_table[p_value_column].le(5E-8)

common_marker_table[cgws_mask].to_csv('CommonGWS_markers.txt', sep='\t')
betas_slice = common_marker_table[['CP']+beta_columns].copy()[cgws_mask]
betas_slice = betas_slice.dropna(axis=0)
betas_slice = betas_slice.set_index('CP')
betas_slice = betas_slice.sort_values(by=['CP'],ascending=False)


def plot_lines_linking_genomic_coordinates2heatmap(bins, coordinates):
    #bins is the number of bins in the heatmap
    #coordinates are the coordinates in ascending order
    num_coords = len(coordinates)
    print(num_coords)
    start = coordinates[0]
    end = coordinates[num_coords-1]
    print(start)
    print(end)
    span = end - start
    print(span)
    # = (span / bins) / 2
    midpoint = span / bins
    #heatmap_coords = [start]+[start + (x*midpoint) for x in range(1,num_coords)]
    heatmap_coords = [start+midpoint]+[start + (x*midpoint + midpoint) for x in range(1,num_coords)]
    #heatmap_coords = [start]+[(x*midpoint + midpoint) for x in range(1,num_coords)]
    print(heatmap_coords)
    sns.set_theme(style="white", palette=None)
    plt.clf()
    fig = plt.figure(figsize=(50,25))
    ax = fig.add_subplot(111)
    #make a straight line at the genomic position
    xs1 = np.array([coordinates, coordinates])
    ys1 = np.array([0, 1])
    plt.plot(xs1, ys1, color='k', linewidth=10)
    #make diagonal lines from genomic position to midpoint
    xs2 = np.array([coordinates, heatmap_coords])
    ys2 = np.array([1, 2])
    plt.plot(xs2, ys2, color='k', linewidth=10)
    #make straight lines from midpoint to midpoint
    xs3 = np.array([heatmap_coords, heatmap_coords])
    ys3 = np.array([2, 3])
    plt.plot(xs3, ys3, color='k', linewidth=10)
    ax.axes.yaxis.set_visible(False)
    ax.axes.xaxis.set_visible(False)
    plt.axis('off')
    plt.savefig(version+'points.png')

cps = betas_slice.index.to_list()
pos = [int(x.split(":")[1]) for x in cps]
pos.reverse()
# = betas_slice.index.str.split(":")
#points = np.array([10877617,11076904,11613715])
hm_columns = len(pos)
plot_lines_linking_genomic_coordinates2heatmap(hm_columns,pos)




'''*****************************************************************************
# XYZ run meta-analyses on LLAS2
removed potential bentham overlaps (alarcon EU samples, criswell EU samples and Vyse EU samples)
will not include HANA from LLAS2 given potential overlap with ALARCON GWAS
will include AAG, AS and EU with the above removed.
altogether:
AAG ~3586
AS ~ 2490
EU ~ 4348

Sumstats should be:
[] 1. AAG
[] 2. AS
[] 3. EU
[] 8. independent_llas2 + metal_wba_lz

*****************************************************************************'''

'''
Setup the metalscript files ...
'''
version = '2022-11-11'
llas2_aag_file = "LLAS2_AAGullah_clec16a.logistic-frq"
llas2_as_file = "LLAS2_AS_clec16a.logistic-frq"
llas2_eu_file = "LLAS2_EU_clec16a.logistic-frq"
metal_wba_file = "metal_wba_lz2022-09-23.txt"
scheme_samplesize = """
#This is sample size weighted std error meta-analysis uses p-value and direction of effect, weighted according to sample size
SCHEME SAMPLESIZE
"""

scheme_stderr = """
#This is inverse variance weighted std error meta-analysis - classical approach, uses effect size estimates and standard errors
SCHEME STDERR
"""

metal_wba_file = """
#Meta-analysis wang, bentham and alarcon SLE GWAS studies
MARKER SNP
ALLELE A1 A2
WEIGHTLABEL NMISS
EFFECT log(OR)
PVALUE P

"""

aag_header = """
#LLAS2 - AAG
MARKER SNP
ALLELE A1 A2
WEIGHTLABEL NMISS
EFFECT log(OR)
STDERRLABEL SE
PVALUE P

"""
aag_process = "PROCESS ./"+llas2_aag_file+"\n"

as_header = """
#LLAS2 - AS
MARKER SNP
ALLELE A1 A2
WEIGHTLABEL NMISS
EFFECT log(OR)
STDERRLABEL SE
PVALUE P

"""
as_process = "PROCESS ./"+llas2_as_file+"\n"


eu_header = """
#LLAS2 - EU
MARKER SNP
ALLELE A1 A2
WEIGHTLABEL NMISS
EFFECT log(OR)
STDERRLABEL SE
PVALUE P

"""
eu_process = "PROCESS ./"+llas2_eu_file+"\n"

#Total n 2489 + 3586 + 4348 = 10423
analyze_quit = """
ANALYZE HETEROGENEITY
QUIT
"""

'''
********************************************************************************
METAL 1 : AAG, AS, EU
********************************************************************************
'''

metal1_header = """
#METAL #1 – LLAS2 study with individuals from these ancestral populations
#LLAS2 - AAG: African American / Gullah
#LLAS2 - AS: East Asian
#LLAS2 – EU: European (Vyse, Criswell and Alarcon samples removed)
"""

metalfile1 = "llas2-clec16a-"+version+".txt"
outfile1 = "OUTFILE META_llas2_clec16a .tbl\n"


metalscript1 = metal1_header + scheme_samplesize + \
aag_header + aag_process + \
as_header + as_process + \
eu_header + eu_process + \
outfile1 + analyze_quit

'''
metalscript1 = metal1_header + scheme_stderr + \
aag_header + aag_process + \
as_header + as_process + \
eu_header + eu_process + \
outfile1 + analyze_quit
'''

with open(metalfile1, 'w') as f:
    f.write(metalscript1)
    f.close()

metal = "./metal"
os.system(metal + " < " + metalfile1)

'''
********************************************************************************
SETUP METAL 2 : +wang, +bentham +alarcon
********************************************************************************
'''


'''
Setup the metalscript files ...
Note: slightly changed from above – loading based on SNP
'''

scheme_samplesize = """
#This is sample size weighted std error meta-analysis uses p-value and direction of effect, weighted according to sample size
SCHEME SAMPLESIZE
"""

scheme_stderr = """
#This is inverse variance weighted std error meta-analysis - classical approach, uses effect size estimates and standard errors
SCHEME STDERR
"""

bentham_metal_header = """
#Bentham et al. SLE GWAS
MARKER SNP
ALLELE A1 A2
EFFECT BetaBentham
STDERRLABEL SEBentham
PVAL PBentham
DEFAULTWEIGHT 10995
ADDFILTER SNP IN (rs9286879,rs917997,rs17810546,rs10045431,rs7746082,rs2301436,rs3764147,rs17228212,rs725613,rs17673553,rs991804,rs744166,rs763361,rs4807569,rs762421)
"""
bentham_metal_process = "PROCESS ../2022-09-23/bentham_metal_format2022-09-23.txt\n"

wang_metal_header = """
#Wang et al. SLE GWAS
MARKER SNP
ALLELE A1 A2
EFFECT BetaWang
STDERRLABEL SEWang
PVAL PWang
DEFAULTWEIGHT 12653
ADDFILTER SNP IN (rs9286879,rs917997,rs17810546,rs10045431,rs7746082,rs2301436,rs3764147,rs17228212,rs725613,rs17673553,rs991804,rs744166,rs763361,rs4807569,rs762421)
"""
wang_metal_process = "PROCESS ../2022-09-23/wang_metal_format2022-09-23.txt\n"

alarcon_metal_header = """
#Alarcon et al. SLE GWAS
MARKER SNP
ALLELE A1 A2
EFFECT BetaAlarcon
STDERRLABEL SEAlarcon
PVAL PAlarcon
DEFAULTWEIGHT 2279
ADDFILTER SNP IN (rs9286879,rs917997,rs17810546,rs10045431,rs7746082,rs2301436,rs3764147,rs17228212,rs725613,rs17673553,rs991804,rs744166,rs763361,rs4807569,rs762421)
"""
alarcon_metal_process = "PROCESS ../2022-09-23/alarcon_metal_format2022-09-23.txt\n"

#Total n 37026 = 7500 + ~29526
analyze_quit = """
ANALYZE HETEROGENEITY
QUIT
"""

'''
********************************************************************************
METAL 2 : Bentham, Wang, Alarcon, LLAS2:{AAG, AS, EU}
********************************************************************************
'''

metal2_header = """
#METAL #2 – Bentham, Wang, Alarcon
#EAvyse_HK_HS_imputed_2metal_6_24_2022.txt + independent LLAS2 samples
#EAvyse_HK_HS_imputed_2metal_6_24_2022.txt processed into individual files
#by CLEC16Av3.py
"""

metalfile2 = "bentham-wang-alarcon-llas2"+version+".txt"
outfile2 = "OUTFILE META_wba-llas2_SampleSize .tbl\n"


metalscript2 = metal2_header + scheme_samplesize + \
bentham_metal_header + bentham_metal_process + \
wang_metal_header + wang_metal_process + \
alarcon_metal_header + alarcon_metal_process + \
aag_header + aag_process + \
as_header + as_process + \
eu_header + eu_process + \
outfile2 + analyze_quit

'''
metalscript2 = metal2_header + scheme_stderr + \
bentham_metal_header + bentham_metal_process + \
wang_metal_header + wang_metal_process + \
alarcon_metal_header + alarcon_metal_process + \
aag_header + aag_process + \
as_header + as_process + \
eu_header + eu_process + \
outfile2 + analyze_quit
'''

with open(metalfile2, 'w') as f:
    f.write(metalscript2)
    f.close()

metal = "./metal"
os.system(metal + " < " + metalfile2)

'''
set up output columns
'''
oc_metal =['MarkerName', 'P']
oc_gwas = ['CPA2A1', 'P', 'Beta', 'SE', 'F_U']

'''
********************************************************************************
clinical manifestations : All of LLAS2:{AAG, AS, EU, HANA}
see 20202-10-05 Analysis file for setup details...
********************************************************************************
'''

#adapted from aa sle gwas : clinical-sumsv3.py
import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency

df = pd.read_csv("rs725613_CaseOnly_Lineup.txt", sep='\t')

df.columns = ['FID-IID', 'Ancestry', 'FID', 'IID', 'PID', 'MID', 'Sex', 'Phenotype', 'rs725613_A_C', 'rs725613_A_C.1', 'rs725613GT', 'rs725613_A', 'rs725613_C', 'rs725613_AA', 'rs725613_AC', 'rs725613_CC', 'GT_In_Clinical', 'Match', 'Clinical_In_GT', 'FID-IID_clin', 'FID_clin', 'IID_clin', 'Contributor', 'Analysis_Sex_1_male_2_female_3_miss', 'race_analysis', 'analysis_status', 'OMRF_Person ID', 'sle_status', 'yob', 'age_of_onset', 'year_sample_collection', 'state_of_residence', 'malar', 'discoid', 'photosensitivity', 'oral_ulcers', 'arthritis', 'serositis', 'pleurisy', 'pericarditis', 'renal', 'proteinuria', 'cellular_casts', 'neurologic', 'seizures', 'psychosis', 'hematology', 'hemolytic_anemia', 'thrombocytopenia', 'leukopenia', 'lymphopenia', 'immunologic', 'le_cell', 'false_positive_vdrl', 'lupus_anticoagulant', 'ana', 'ds_dna', 'anti_sm', 'anti_ro', 'anti_la', 'anti_nrnp', 'anti_p', 'ana_titer_binary', 'ana_titer_quantitative', 'rheumatoid_factor_binary', 'rheumatoid_factor_quantitative', 'igg_binary', 'igm_binary', 'ch_50_binary', 'ch_50_percentLLN_quantitative', 'C3_Low_Binary', 'C3_percentLLN_quantitative', 'C4_low_binary']

#A is the risk allele.
#df.rs725613GT
#df.rs725613_A

#replace -9 with nan
binary_columns = ['malar', 'discoid', 'photosensitivity', 'oral_ulcers', 'arthritis', 'serositis', 'pleurisy', 'pericarditis', 'renal', 'proteinuria', 'cellular_casts', 'neurologic', 'seizures', 'psychosis', 'hematology', 'hemolytic_anemia', 'thrombocytopenia', 'leukopenia', 'lymphopenia', 'immunologic', 'le_cell', 'false_positive_vdrl', 'lupus_anticoagulant', 'ana', 'ds_dna', 'anti_sm', 'anti_ro', 'anti_la', 'anti_nrnp', 'anti_p', 'ana_titer_binary', 'rheumatoid_factor_binary', 'igg_binary', 'igm_binary', 'ch_50_binary', 'C3_Low_Binary', 'C4_low_binary']

df[binary_columns] = df[binary_columns].replace(-9, np.nan)

df[['ana', 'ds_dna', 'anti_sm', 'anti_ro', 'anti_la', 'anti_nrnp', 'anti_p']].describe()

ct = pd.crosstab(df.rs725613GT, df.renal, margins=True)

value = np.array([ct.iloc[0][0:2].values,
                  ct.iloc[1][0:2].values,
                  ct.iloc[2][0:2].values])

#column_list = ['malar', 'discoid', 'photosensitivity', 'oral_ulcers', 'arthritis', 'serositis', 'pleurisy', 'pericarditis', 'renal', 'proteinuria', 'cellular_casts', 'neurologic', 'seizures', 'psychosis', 'hematology', 'hemolytic_anemia', 'thrombocytopenia', 'leukopenia', 'lymphopenia', 'immunologic', 'le_cell', 'false_positive_vdrl', 'lupus_anticoagulant', 'ana', 'ds_dna', 'anti_sm', 'anti_ro', 'anti_la', 'anti_nrnp', 'anti_p']
for column in binary_columns:
    print(column)
    print("2X3 table")
    #drop_na = df[column].dropna(axis=1)
    ct = pd.crosstab(df[column], df.rs725613GT, margins=True)
    print(ct)
    #print(ct.iloc[0][0:2].values)
    value = np.array([ct.iloc[0][0:3].values, ct.iloc[1][0:3].values])
    print(value)
    print(" column chi squared: (Statistic, P-value, d.f.)")
    chi, p, degf, expected = chi2_contingency(value)
    print("("+str(chi)+", " +str(p) +", "+str(degf)+")")
    std_residuals = (value-expected)/(np.sqrt(expected))
    print("standardized residuals: ")
    print(std_residuals)
    print("total")
    print(np.sum(value))
    print("%observed")
    print(value/np.sum(value))
    print("%expected")
    print(expected/np.sum(value))

#Now double count homozygotes for 2X2 tables
for column in binary_columns:
    print(column)
    print("2X2 table")
    ct = pd.crosstab(df[column], df.rs725613GT, margins=True)
    print(ct)
    protective_n = (ct.iloc[0][0] * 2) + ct.iloc[0][1]
    protective_y = (ct.iloc[1][0] * 2) + ct.iloc[1][1]
    risk_n = ct.iloc[0][1] + (ct.iloc[0][2] * 2)
    risk_y = ct.iloc[1][1] + (ct.iloc[1][2] * 2)
    no_array = np.array([protective_n,risk_n])
    yes_array = np.array([protective_y, risk_y])
    print(no_array[0:2])
    print(yes_array[0:2])
    value = np.array([no_array[0:2], yes_array[0:2]])
    print(value)
    print(" column chi squared: (Statistic, P-value, d.f.)")
    chi, p, degf, expected = chi2_contingency(value)
    print("("+str(chi)+", " +str(p) +", "+str(degf)+")")
    std_residuals = (value-expected)/(np.sqrt(expected))
    print("standardized residuals: ")
    print(std_residuals)
    print("total")
    print(np.sum(value))
    print("%observed")
    print(value/np.sum(value))
    print("%expected")
    print(expected/np.sum(value))


#'C3_Low_Binary' removed – causes traceback in asian samples due to too little data in the cross tab. numbers are too small
binary_columns = ['malar', 'discoid', 'photosensitivity', 'oral_ulcers', 'arthritis', 'serositis', 'pleurisy', 'pericarditis', 'renal', 'proteinuria', 'cellular_casts', 'neurologic', 'seizures', 'psychosis', 'hematology', 'hemolytic_anemia', 'thrombocytopenia', 'leukopenia', 'lymphopenia', 'immunologic', 'le_cell', 'false_positive_vdrl', 'lupus_anticoagulant', 'ana', 'ds_dna', 'anti_sm', 'anti_ro', 'anti_la', 'anti_nrnp', 'anti_p', 'ana_titer_binary', 'rheumatoid_factor_binary', 'igg_binary', 'igm_binary', 'ch_50_binary', 'C4_low_binary']

ancestries = df.groupby('Ancestry')
for key, ancestry in ancestries:
    stars = "******************************************************************"
    print(stars+str(key)+stars)
    for column in binary_columns:
        print(str(column)+": 2X3 table " + str(key))
        ct = pd.crosstab(ancestry[column], ancestry.rs725613GT, margins=True)
        print(ct)
        value = np.array([ct.iloc[0][0:3].values, ct.iloc[1][0:3].values])
        #print(value)
        print(" column chi squared: (Statistic, P-value, d.f.)")
        chi, p, degf, expected = chi2_contingency(value)
        print("("+str(chi)+", " +str(p) +", "+str(degf)+")")
        std_residuals = (value-expected)/(np.sqrt(expected))
        print("standardized residuals: ")
        print(std_residuals)


'''*****************************************************************************
# QRS run meta-analyses on "seropositive" autoimmune diseases:
T1D, MS, PBC, PSC and SLE
all are gwas except onengut_t1d
*****************************************************************************'''

'''
Setup the metalscript files ...
'''
version = '2022-11-11'

ji_psc_file = "ji_psc_gwas_metal-2022-10-07.txt"
cordell_pbc_file = "cordell_pbc_gwas_metal-2022-10-07.txt"
sawcer_ms_file = "sawcer_ms_gwas_metal-2022-10-07.txt"
bentham_sle_file = "bentham_sle_gwas_metal-2022-10-07-v2-.txt"
onengut_t1d_file = "onengut_t1d_gwas_metal-2022-10-07.txt"

scheme_samplesize = """
#This is sample size weighted std error meta-analysis uses p-value and direction of effect, weighted according to sample size
SCHEME SAMPLESIZE
"""

scheme_stderr = """
#This is inverse variance weighted std error meta-analysis - classical approach, uses effect size estimates and standard errors
SCHEME STDERR
"""

'''
for the uploads to locus zoom, the following parameters were selected:
    chr:hm_chrom
    pos:hm_pos
    ref allele: hm_other_allele
    alt allele: hm_effect_allele
    p-value: p_value
'''

metal_jobcs_header = """
#Meta-analysis ji, onengut (IC), bentham, cordell and sawcer GWAS studies
MARKER hm_variant_id
ALLELE hm_other_allele hm_effect_allele
EFFECT hm_beta
STDERRLABEL standard_error
PVALUE p_value

"""

ji_psc_header = """
#ji_psc
DEFAULTWEIGHT 14890
"""
ji_psc_process = "PROCESS ./"+ji_psc_file+"\n"

cordell_pbc_header = """
#cordell_pbc
DEFAULTWEIGHT 13239
"""
cordell_pbc_process = "PROCESS ./"+cordell_pbc_file+"\n"

sawcer_ms_header = """
#sawcer_ms
DEFAULTWEIGHT 26621
"""
sawcer_ms_process = "PROCESS ./"+sawcer_ms_file+"\n"

bentham_sle_header = """
#bentham_sle
DEFAULTWEIGHT 10995
"""
bentham_sle_process = "PROCESS ./"+bentham_sle_file+"\n"

onengut_t1d_header = """
#onengut_t1d
DEFAULTWEIGHT 29652
"""
onengut_t1d_process = "PROCESS ./"+onengut_t1d_file+"\n"

#Total n = 95397
analyze_quit = """
ANALYZE HETEROGENEITY
QUIT
"""

'''
********************************************************************************
METAL 1 : MS, PBC, PSC, SLE and T1D
********************************************************************************
'''

metal1_header = """
#METAL – Meta-analysis ji, onengut (IC), bentham, cordell and sawcer GWAS studies
"""

metalfile1 = "seropositive-metal-"+version+".txt"
outfile1 = "OUTFILE META_seropositive_ms_psc_sle_t1d_pbc .tbl\n"

metalscript1 = metal1_header + metal_jobcs_header + scheme_samplesize + \
ji_psc_header + ji_psc_process + \
cordell_pbc_header + cordell_pbc_process + \
sawcer_ms_header + sawcer_ms_process + \
bentham_sle_header + bentham_sle_process + \
onengut_t1d_header + onengut_t1d_process + \
outfile1 + analyze_quit

with open(metalfile1, 'w') as f:
    f.write(metalscript1)
    f.close()

metal = "./metal"
os.system(metal + " < " + metalfile1)

'''
********************************************************************************
METAL 2 : MS, PBC, PSC and SLE
********************************************************************************
'''

metal2_header = """
#METAL – Meta-analysis: ji, bentham, cordell and sawcer GWAS studies
"""

metalfile2 = "seropositive-metal-"+version+".txt"
outfile2 = "OUTFILE META_seropositive_ms_psc_sle_pbc .tbl\n"

metalscript2 = metal2_header + metal_jobcs_header + scheme_samplesize + \
ji_psc_header + ji_psc_process + \
cordell_pbc_header + cordell_pbc_process + \
sawcer_ms_header + sawcer_ms_process + \
bentham_sle_header + bentham_sle_process + \
outfile2 + analyze_quit

with open(metalfile2, 'w') as f:
    f.write(metalscript2)
    f.close()

metal = "./metal"
os.system(metal + " < " + metalfile2)


'''
********************************************************************************
 METAL 1 : MS, PBC, PSC, SLE and T1D – locuszoom prep
********************************************************************************
'''
''' METAL : MS, PBC, PSC, SLE and T1D '''
base6 = "META_seropositive_ms_psc_sle_t1d_pbc1"
metal_seropos = pd.read_csv(base6 +".tbl", lineterminator='\n', delim_whitespace=True)

metal_seropos[['CHR', 'POS']] = metal_seropos['MarkerName'].str.split(r":|_", 2, expand=True)[[0,1]]

remap_dict = {'X':23}
metal_seropos['CHR'] = metal_seropos['CHR'].replace(remap_dict)
metal_seropos = metal_seropos.astype({'CHR' : 'int32', 'POS' : 'int32'})

metal_seropos.columns = ['MarkerName', 'Allele1', 'Allele2', 'Weight', 'Zscore', 'P', 'Direction', 'HetISq', 'HetChiSq', 'HetDf', 'HetPVal', 'CHR', 'POS']

metal_seropos = metal_seropos.sort_values(by=['CHR', 'POS'])

metal_seropos.equals(metal_seropos.sort_values(by=['CHR', 'POS']))

oc_metal =['CHR', 'POS', 'Allele2', 'Allele1', 'P']

metal_seropos[oc_metal].to_csv('metal_ms_psc_sle_t1d_pbc1_lz'+version+'.txt', sep='\t', index=False)
os.system("gzip -k " + 'metal_ms_psc_sle_t1d_pbc1_lz'+version+'.txt &')

#oc_metal =['MarkerName', 'P']
#oc_gwas = ['CPA2A1', 'P', 'Beta', 'SE', 'F_U']

'''
********************************************************************************
 METAL 2 : MS, PBC, PSC and SLE  – locuszoom prep
********************************************************************************
'''
''' METAL : MS, PBC, PSC and SLE  '''
base7 = "META_seropositive_ms_psc_sle_pbc1"
metal_seropos2 = pd.read_csv(base7 +".tbl", lineterminator='\n', delim_whitespace=True)

metal_seropos2[['CHR', 'POS']] = metal_seropos2['MarkerName'].str.split(r":|_", 2, expand=True)[[0,1]]

remap_dict = {'X':23}
metal_seropos2['CHR'] = metal_seropos2['CHR'].replace(remap_dict)
metal_seropos2 = metal_seropos2.astype({'CHR' : 'int32', 'POS' : 'int32'})

metal_seropos2.columns = ['MarkerName', 'Allele1', 'Allele2', 'Weight', 'Zscore', 'P', 'Direction', 'HetISq', 'HetChiSq', 'HetDf', 'HetPVal', 'CHR', 'POS']

metal_seropos2 = metal_seropos2.sort_values(by=['CHR', 'POS'])

metal_seropos2.equals(metal_seropos2.sort_values(by=['CHR', 'POS']))

oc_metal =['CHR', 'POS', 'Allele2', 'Allele1', 'P']

metal_seropos2[oc_metal].to_csv('metal_ms_psc_sle_pbc1_lz'+version+'.txt', sep='\t', index=False)
os.system("gzip -k " + 'metal_ms_psc_sle_pbc1_lz'+version+'.txt &')

'''
bentham ballons
version = '2022-10-07-v2-'
'''
import sys
print(sys.version)
print(pd.__version__)
print(np.__version__)
import matplotlib
print(matplotlib.__version__)
import scipy
print(scipy.__version__)
print(sns.__version__)
