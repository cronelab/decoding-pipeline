import collections



############################# ANATOMICAL CHANNELS #############################

# Initializing the dictionary for patient channels.
patient_channels = collections.defaultdict(dict)

# PY15N001
patient_channels['PY15N001']['motor']   = ['HDG1', 'HDG2','HDG3','HDG4','HDG5','HDG6','HDG7','HDG8','HDG9','HDG10','HDG11','HDG12','HDG13','HDG14','HDG15','HDG16',\
                                           'HDG17','HDG18','HDG19','HDG20', 'HDG21','HDG22','HDG23','HDG24','HDG25','HDG26','HDG27','HDG28','HDG29','HDG30','HDG31',\
                                           'HDG32','HDG33','HDG34','HDG35','HDG36','HDG37','HDG38','HDG39', 'HDG40','HDG41','HDG42','HDG43','HDG44','HDG45','HDG46',\
                                           'HDG47','HDG48','HDG51','HDG52']
patient_channels['PY15N001']['sensory'] = ['HDG49','HDG50','HDG53','HDG54','HDG55','HDG56','HDG57','HDG58','HDG59','HDG60','HDG61','HDG62','HDG63','HDG64','HDG65',\
                                           'HDG66','HDG67','HDG68','HDG69','HDG70','HDG71','HDG72','HDG73','HDG74','HDG75','HDG76','HDG77','HDG78','HDG79','HDG80',\
                                           'HDG81','HDG82','HDG83','HDG84','HDG85','HDG86','HDG87','HDG88', 'HDG89','HDG90','HDG91','HDG92','HDG93','HDG94','HDG95',\
                                           'HDG96','HDG97','HDG98','HDG99','HDG100','HDG101','HDG102','HDG103','HDG104','HDG105','HDG106','HDG107','HDG108','HDG109',\
                                           'HDG110','HDG111','HDG112','HDG113','HDG114','HDG115','HDG116','HDG117','HDG118','HDG119','HDG120','HDG121','HDG122','HDG123',\
                                           'HDG124','HDG125','HDG126','HDG127','HDG128']

# PY17N009
patient_channels['PY17N009']['motor']   = []
patient_channels['PY17N009']['sensory'] = []

# PY17N010
patient_channels['PY17N010']['motor']   = ['RFG14','RFG15','RFG16','RFG22','RFG23','RFG24','RFG30','RFG31','RFG32','RFG37','RFG38','RFG39','RFG40','RFG44','RFG45',\
                                           'RFG46','RFG47','RFG48','RFG50', 'RFG51','RFG52','RFG53','RFG54','RFG55','RFG56','RFG58','RFG59','RFG60','RFG61','RFG62',\
                                           'RFG63','RFG64','AMIC1','AMIC2','AMIC3','AMIC4','AMIC5','AMIC6', 'AMIC7','AMIC8','AMIC9','AMIC10','AMIC11','AMIC12','AMIC13',\
                                           'AMIC14','AMIC15','AMIC16','PMIC1','PMIC2','PMIC3','PMIC4','PMIC5','PMIC6','PMIC7','PMIC8','PMIC9','PMIC10','PMIC11','PMIC12',\
                                           'PMIC13','PMIC14','PMIC15','PMIC16']
patient_channels['PY17N010']['sensory'] = ['RFG12','RFG13','RFG20','RFG21','RFG28','RFG29','RFG35','RFG36','RFG42','RFG43','RFG49','RFG57']

# PY17N013
patient_channels['PY17N013']['motor']   = ['LFPG17','LFPG18','LFPG19','LFPG20','LFPG21','LFPG22','LFPG23','LFPG24','LFPG25','LFPG26','LFPG27','LFPG28','LFPG29','LFPG30',\
                                           'LFPG31','LFPG33','LFPG34', 'LFPG35','LFPG36','LFPG37','LFPG38','LFPG38','LFPG40','LFPG41','LFPG42','LFPG43','LFPG44',\
                                           'LFPG45', 'LFPG46','LFPG47','LFPG49','LFPG50','LFPG51','LFPG52', 'LFPG53','LFPG54','LFPG55','LFPG56','LFPG57','LFPG58',\
                                           'LFPG59','LFPG46','LFPG65','LFPG66','LFPG67','LFPG71','LFPG78','LFPG79','LSPS6']
patient_channels['PY17N013']['sensory'] = ['LFPG68','LFPG69','LFPG70','LFPG72','LFPG73','LFPG74','LFPG75','LFPG77','LFPG81','LFPG82','LFPG83','LFPG84','LFPG85','LFPG86',\
                                           'LFPG87','LFPG88','LFPG89','LFPG90','LFPG91','LFPG92','LFPG93','LFPG94','LFPG97','LFPG98','LFPG99','LFPG100','LFPG101',\
                                           'LFPG102','LFPG103','LFPG104','LFPG105','LFPG106','LFPG107','LFPG108','LFPG109','LFPG110','LFPG113','LFPG114','LFPG115',\
                                           'LFPG116','LFPG117','LFPG118','LFPG119','LFPG120','LFPG121','LFPG122','LFPG123','LFPG124','LFPG125','LSPS5','LSPS4']

# CC01
patient_channels['CC01']['motor']       = ['chan66','chan67','chan68','chan69','chan70','chan74','chan75','chan76','chan77','chan78','chan84','chan85','chan86','chan91',\
                                           'chan92','chan93','chan94', 'chan99','chan100','chan101','chan102','chan108','chan109','chan110','chan117','chan118',\
                                           'chan125','chan126']
patient_channels['CC01']['sensory']     = ['chan71','chan72','chan79','chan80','chan87','chan88','chan95','chan96','chan103','chan104','chan112','chan120','chan128']





############################# BAD CHANNELS #############################

# Initializing the dictionary for channels to be eliminated from future analysis. These are in addition to previously recorded bad channels. 
elim_channels = {}

# PY15N001
elim_channels['PY15N001'] = ['ainp1', 'ainp2', 'ainp3', 'ainp4', 'ainp5', 'ainp6', 'ainp7', 'ainp8', 'ainp9', 'ainp10', 'ainp11', 'ainp12', 'ainp13', 'ainp14', 'ainp15',\
                             'ainp16', 'ainp17', 'ainp18', 'ainp19', 'ainp20', 'ainp21', 'ainp22', 'ainp23', 'ainp24', 'ainp25', 'ainp26', 'ainp27', 'ainp28', 'ainp29',\
                             'ainp30', 'ainp31', 'ainp32', 'HDG7', 'HDG35', 'HDG50','ODA1', 'ODA2', 'ODA3', 'ODA4', 'ODA5', 'ODA6', 'ODA7', 'ODA8', 'ODB1', 'ODB2',\
                             'ODB3', 'ODB4', 'ODB5', 'ODB6', 'ODB7', 'ODB8', 'ODC1', 'ODC2', 'ODC3', 'ODC4', 'ODC5', 'ODC6', 'ODC7', 'ODC8', 'ODD1', 'ODD2', 'ODD3',\
                             'ODD4', 'ODD5', 'ODD6', 'ODD7', 'ODD8', 'DPS2','SKIP'] 

# PY17N009
elim_channels['PY17N009'] = ['LMPS1', 'LMPS2', 'LMPS3', 'LMPS4', 'REF1', 'REF2', 'LPPS1', 'LPPS2', 'LPPS3', 'LPPS4', 'LCPS1', 'LCPS2', 'LCPS3', 'LCPS4', 'LCPS5',\
                             'LCPS6', 'LCPS7', 'LCPS8', 'LCPS9', 'LCPS10', 'LCPS11', 'LCPS12', 'LCPS13', 'LCPS14', 'LCPS15', 'LCPS16', 'LFS17', 'LFS18', 'LFS19', 'LFS20',\
                             'LFS21', 'LFS22', 'LFS23', 'LFS24', 'LFS25', 'LFS26', 'LFS27', 'LFS28', 'LFS29', 'LFS30', 'LFS31', 'LFS32', 'LSPS1', 'LSPS2', 'LSPS3',\
                             'LSPS4', 'LSPS5', 'LSPS6', 'EKG1', 'EKG2', 'EMG1', 'EMG2', 'EMG3', 'EMG4', 'EMG5', 'EMG6', 'EMG7', 'EMG8', 'EMG9', 'EMG10', 'EMG11', 'EMG12',\
                             'EMG13', 'EMG14']

# PY17N010
elim_channels['PY17N010'] = ['ainp1','ainp2','ainp3','ainp4','ainp16']

# PY17N013
elim_channels['PY17N013'] = ['EKG1', 'EKG2', 'EMG1', 'EMG2', 'EMG3', 'EMG4', 'EMG5', 'EMG6', 'EMG7', 'EMG8', 'EMG9', 'EMG10', 'EMG11', 'EMG12', 'EMG13', 'EMG14', 'DC01',\
                             'DC02', 'DC03', 'DC04', 'DC05', 'DC06', 'DC07', 'DC08', 'DC09', 'DC10', 'DC11', 'DC12', 'DC13', 'DC14', 'DC15', 'DC16', 'LFPG120', 'LFPG121',\
                             'LFPG122', 'LFPG123', 'LFPG124', 'LFPG125', 'LFPG126', 'LFPG127', 'LFPG128', 'REF1', 'REF2']

# CC01
elim_channels['CC01']     = ['ainp1','ainp2','ainp3']




############################# CAR? #############################

# Initializing the dictionary of whether the CAR-ed signals will be used for each subject.
car = {}

car['PY15N001'] = 'Yes'
car['PY17N009'] = 'No'
car['PY17N010'] = 'Yes'
car['PY17N013'] = 'No'
car['CC01']     = 'No'


############################# CAR CHANNELS #############################

# Initializing the dictionary for each subject, which holds the sets of channels to be independently CAR-ed.
car_channels = {}

# PY15N001
car_channels['PY15N001'] = [['HDG33','HDG34','HDG36','HDG39','HDG40','HDG41','HDG42','HDG43','HDG44','HDG45','HDG46','HDG47','HDG48','HDG49','HDG51','HDG52']]

# PY17N009
car_channels['PY17N009'] = []

# PY17N010
car_channels['PY17N010'] = []

# PY17N013
car_channels['PY17N013'] = []

# CC01
car_channels['CC01'] = []



