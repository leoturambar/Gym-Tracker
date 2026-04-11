# Gruppi muscolari
MUSCLES = ['Petto', 'Spalle', 'Tricipiti', 'Quadricipiti', 'Core', 'Femorali', 'Schiena', 'Bicipiti']

# Tipi di esercizio
# weighted     → peso esterno (kg)
# bodyweight   → solo peso corporeo, RTV = 1.0
# weighted_bw  → peso corporeo + peso aggiunto (dips, trazioni zavorate)
# timed        → durata in secondi, RTV = durata / 120
# excluded     → non contato (stretching, attivazione)

DAYS = [
    {
        'id': 1,
        'name': 'Upper 1',
        'sub': 'Petto + Tricipiti',
        'exercises': [
            {'name': 'Band pull apart',    'type': 'excluded'},
            {'name': 'Chest press',        'type': 'weighted',    'default': 40},
            {'name': 'Pec fly',            'type': 'weighted',    'default': 20},
            {'name': 'Dips',               'type': 'weighted_bw', 'default': -10},
            {'name': 'Tricipiti ai cavi',  'type': 'weighted',    'default': 10},
            {'name': 'Shoulder press',     'type': 'weighted',    'default': 25},
            {'name': 'Lat machine',        'type': 'weighted',    'default': 35},
            {'name': 'Delts machine',      'type': 'weighted',    'default': 15},
            {'name': 'Plank',              'type': 'timed',       'default': 60},
        ]
    },
    {
        'id': 2,
        'name': 'Lower 1',
        'sub': 'Quad + Core',
        'exercises': [
            {'name': 'Leg press (piedi medi)',  'type': 'weighted', 'default': 50},
            {'name': 'Leg extension',           'type': 'weighted', 'default': 20},
            {'name': 'Glute machine',           'type': 'bodyweight'},
            {'name': 'Adduttori',               'type': 'weighted', 'default': 35},
            {'name': 'Abduttori',               'type': 'weighted', 'default': 35},
            {'name': 'Calf raise',              'type': 'weighted', 'default': 35},
            {'name': 'Addominali laterali',     'type': 'weighted', 'default': 5},
            {'name': 'Abs machine',             'type': 'weighted', 'default': 32},
            {'name': 'Hanging leg raise',       'type': 'bodyweight'},
            {'name': 'Plank con rotazione',     'type': 'timed',    'default': 60},
        ]
    },
    {
        'id': 3,
        'name': 'Upper 2',
        'sub': 'Schiena + Bicipiti',
        'exercises': [
            {'name': 'Scapular pull up',    'type': 'bodyweight'},
            {'name': 'Trazioni',            'type': 'weighted_bw', 'default': -15},
            {'name': 'Rematore',            'type': 'weighted',    'default': 35},
            {'name': 'Pulley basso',        'type': 'weighted',    'default': 35},
            {'name': 'Curl',                'type': 'weighted',    'default': 15},
            {'name': 'Chest press',         'type': 'weighted',    'default': 40},
            {'name': 'Delts machine',       'type': 'weighted',    'default': 15},
            {'name': 'Plank',               'type': 'timed',       'default': 120},
            {'name': 'Plank laterale',      'type': 'timed',       'default': 120},
        ]
    },
    {
        'id': 4,
        'name': 'Lower 2',
        'sub': 'Posteriore + Core',
        'exercises': [
            {'name': 'Leg curl',                     'type': 'weighted', 'default': 32},
            {'name': 'Leg press (piedi alti/larghi)', 'type': 'weighted', 'default': 50},
            {'name': 'Iperestensioni lombari',        'type': 'weighted', 'default': 10},
            {'name': 'Adduttori',                     'type': 'weighted', 'default': 35},
            {'name': 'Abduttori',                     'type': 'weighted', 'default': 35},
            {'name': 'Calf raise',                    'type': 'weighted', 'default': 35},
            {'name': 'Abs machine',                   'type': 'weighted', 'default': 32},
            {'name': 'Hanging knee raise',            'type': 'bodyweight'},
            {'name': 'Russian twists',                'type': 'weighted', 'default': 6},
        ]
    },
]

# Mappatura esercizio → gruppi muscolari stimolati
EX_MUSCLES = {
    'Chest press':                  ['Petto', 'Tricipiti'],
    'Pec fly':                      ['Petto'],
    'Dips':                         ['Tricipiti', 'Petto'],
    'Tricipiti ai cavi':            ['Tricipiti'],
    'Shoulder press':               ['Spalle', 'Tricipiti'],
    'Lat machine':                  ['Schiena', 'Bicipiti'],
    'Delts machine':                ['Spalle'],
    'Plank':                        ['Core'],
    'Leg press (piedi medi)':       ['Quadricipiti'],
    'Leg extension':                ['Quadricipiti'],
    'Glute machine':                ['Femorali'],
    'Adduttori':                    ['Quadricipiti'],
    'Abduttori':                    ['Femorali'],
    'Calf raise':                   ['Quadricipiti'],
    'Addominali laterali':          ['Core'],
    'Abs machine':                  ['Core'],
    'Hanging leg raise':            ['Core'],
    'Plank con rotazione':          ['Core'],
    'Scapular pull up':             ['Schiena', 'Spalle'],
    'Trazioni':                     ['Schiena', 'Bicipiti'],
    'Rematore':                     ['Schiena', 'Bicipiti'],
    'Pulley basso':                 ['Schiena', 'Bicipiti'],
    'Curl':                         ['Bicipiti'],
    'Plank laterale':               ['Core'],
    'Leg curl':                     ['Femorali'],
    'Leg press (piedi alti/larghi)':['Femorali', 'Quadricipiti'],
    'Iperestensioni lombari':       ['Femorali', 'Core'],
    'Hanging knee raise':           ['Core'],
    'Russian twists':               ['Core'],
}

# Riferimento temporale per esercizi timed (secondi)
# RTV = durata_effettiva / TIMED_REFERENCE
TIMED_REFERENCE = 120

# Atleta di riferimento — RTV assoluto per gruppo muscolare per settimana completa
# Basato sulla tua scheda, atleta ~75kg che esegue tutti gli esercizi con buoni carichi
REFERENCE_ATHLETE = {
    'Petto':        2.0,   # chest press x2 + dips
    'Spalle':       1.8,   # shoulder press + delts machine x2
    'Tricipiti':    2.2,   # dips + tricipiti cavi + shoulder press x2
    'Quadricipiti': 3.5,   # leg press x2 + leg extension + adduttori x2 + calf x2
    'Femorali':     2.8,   # leg curl + iperestensioni + abduttori x2 + glute
    'Schiena':      3.0,   # trazioni + rematore + pulley + lat machine + scapular
    'Bicipiti':     1.5,   # trazioni + curl + lat machine
    'Core':         3.2,   # plank x4 + hanging x2 + abs x2 + russian + addominali
}