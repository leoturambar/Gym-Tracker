import copy
import json
import os

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
        'name': 'Upper Push',
        'sub': 'Petto + Tricipiti',
        'exercises': [
            {'name': 'Band pull apart',             'type': 'excluded'},
            {'name': 'Face pull cavo',              'type': 'weighted',    'default': 15},
            {'name': 'Incline chest press manubri', 'type': 'weighted',    'default': 20},
            {'name': 'Chest press',                 'type': 'weighted',    'default': 40},
            {'name': 'Shoulder press',              'type': 'weighted',    'default': 25},
            {'name': 'Dips',                        'type': 'weighted_bw', 'default': -10},
            {'name': 'Pec fly',                     'type': 'weighted',    'default': 20},
            {'name': 'Lat machine presa larga',     'type': 'weighted',    'default': 35},
            {'name': 'Delts machine',               'type': 'weighted',    'default': 15},
            {'name': 'Tricipiti ai cavi',           'type': 'weighted',    'default': 10},
            {'name': 'Curl bicipiti cavo',          'type': 'weighted',    'default': 15},  # + pull component
            {'name': 'Hollow body hold',            'type': 'timed',       'default': 45},  # +
            {'name': 'Plank',                       'type': 'timed',       'default': 60},
        ]
    },
    {
        'id': 2,
        'name': 'Lower Quad',
        'sub': 'Quad dominant',
        'exercises': [
            {'name': 'Leg press piedi medi',             'type': 'weighted',    'default': 50},
            {'name': 'Hip thrust',                       'type': 'weighted',    'default': 50},
            {'name': 'Leg extension',                    'type': 'weighted',    'default': 20},
            {'name': 'Glute kickback',                   'type': 'weighted',    'default': 35},  # +
            {'name': 'Adduttori macchina',               'type': 'weighted',    'default': 35},
            {'name': 'Abduttori macchina',               'type': 'weighted',    'default': 35},
            {'name': 'Calf raise',                       'type': 'weighted',    'default': 35},
            {'name': 'Pallof press',                     'type': 'weighted',    'default': 13},  # +
            {'name': 'Addominali laterali panca romana', 'type': 'weighted_bw', 'default': 5},
            {'name': 'Hanging leg raise',                'type': 'bodyweight'},
        ]
    },
    {
        'id': 3,
        'name': 'Upper Pull',
        'sub': 'Schiena + Bicipiti',
        'exercises': [
            {'name': 'Scapular pull up',                          'type': 'excluded'},
            {'name': 'Trazioni assistite',                        'type': 'weighted_bw', 'default': -10},
            {'name': 'Rematore',                                  'type': 'weighted',    'default': 35},
            {'name': 'Pulley basso',                              'type': 'weighted',    'default': 35},
            {'name': 'Face pull cavo',                            'type': 'weighted',    'default': 15},
            {'name': 'Curl bicipiti panca inclinata 60° manubri', 'type': 'weighted',    'default': 15},
            {'name': 'Tricipiti ai cavi',                         'type': 'weighted',    'default': 10},
            {'name': 'Plank',                                     'type': 'timed',       'default': 120},
            {'name': 'Plank laterale',                            'type': 'timed',       'default': 120},
        ]
    },
    {
        'id': 4,
        'name': 'Lower Hip',
        'sub': 'Hamstring dominant',  # fixed typo
        'exercises': [
            {'name': 'Leg curl',                      'type': 'weighted',    'default': 36},
            {'name': 'Leg press piedi alti e larghi', 'type': 'weighted',    'default': 60},
            {'name': 'Iperestensioni lombari',        'type': 'weighted_bw', 'default': 15},
            {'name': 'Romanian deadlift',             'type': 'weighted',    'default': 12},  # +
            {'name': 'Adduttori macchina',            'type': 'weighted',    'default': 35},
            {'name': 'Abduttori macchina',            'type': 'weighted',    'default': 35},
            {'name': 'Calf raise',                    'type': 'weighted',    'default': 35},
            {'name': 'Russian twists',                'type': 'weighted_bw', 'default': 12},
            {'name': 'Abs machine',                   'type': 'weighted',    'default': 32},
            {'name': 'Abs wheel',                     'type': 'bodyweight'},                   # +
            {'name': 'Hanging knee raise',            'type': 'bodyweight'},
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

# ── Exercise metadata store (populated from data/exercises.json) ──────────────

_EXERCISES_PATH = 'data/exercises.json'

_EXERCISE_META: dict = {}

# JSON muscle key → Italian MUSCLES name used throughout the app
_MUSCLE_KEY_MAP = {
    'chest':      'Petto',
    'shoulders':  'Spalle',
    'triceps':    'Tricipiti',
    'back':       'Schiena',
    'biceps':     'Bicipiti',
    'quads':      'Quadricipiti',
    'hamstrings': 'Femorali',
    'core':       'Core',
    'glutes':     'Femorali',    # closest existing group
    'calves':     'Quadricipiti',  # matches existing Calf raise mapping
}

_DAY_ID_MAP = {'D1': 1, 'D2': 2, 'D3': 3, 'D4': 4}

# Pristine copies of the hardcoded fallbacks, used to rebuild on every reload
_DEFAULT_DAYS = copy.deepcopy(DAYS)
_DEFAULT_EX_MUSCLES = copy.deepcopy(EX_MUSCLES)


def get_exercise_meta(name: str) -> dict:
    """Return the full exercises.json metadata dict for an exercise, or {} if not found."""
    return _EXERCISE_META.get(name, {})


def _load_exercise_records() -> list:
    """Read the exercise list from exercises.json (flat-array or dict format)."""
    if not os.path.exists(_EXERCISES_PATH):
        return []
    with open(_EXERCISES_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)
    if isinstance(data, dict):
        data = data.get('exercises', [])
    return data


def reload_exercises():
    """
    Rebuild DAYS, EX_MUSCLES and the metadata store from data/exercises.json.

    Objects are mutated in place, so names imported elsewhere (`from config
    import DAYS`) always see the current state. Call this at the start of every
    Streamlit run so changes written to the JSON (routine sync, new exercises)
    take effect without restarting the app.

    JSON wins on conflict: a hardcoded built-in day exercise is dropped when its
    JSON record exists but no longer lists that day in day_ids.
    """
    DAYS[:] = copy.deepcopy(_DEFAULT_DAYS)
    EX_MUSCLES.clear()
    EX_MUSCLES.update(copy.deepcopy(_DEFAULT_EX_MUSCLES))
    _EXERCISE_META.clear()

    try:
        records = _load_exercise_records()
    except Exception as e:
        print(f"[config] Warning: could not load {_EXERCISES_PATH} — {e}")
        return

    for ex in records:
        name = ex['name']
        _EXERCISE_META[name] = ex

        # Update EX_MUSCLES with mapped Italian muscle names (skip excluded)
        if ex.get('muscles') and ex.get('type') != 'excluded':
            mapped = list(dict.fromkeys(
                _MUSCLE_KEY_MAP[k]
                for k in ex['muscles']
                if k in _MUSCLE_KEY_MAP and _MUSCLE_KEY_MAP[k] in MUSCLES
            ))
            if mapped:
                EX_MUSCLES[name] = mapped

    # Drop hardcoded exercises the JSON has removed from a built-in day
    for day in DAYS:
        did = f"D{day['id']}"
        day['exercises'] = [
            e for e in day['exercises']
            if e['name'] not in _EXERCISE_META
            or did in _EXERCISE_META[e['name']].get('day_ids', [])
        ]

    # Update or append each JSON exercise in its target days
    for ex in records:
        name = ex['name']
        for did in ex.get('day_ids', []):
            day_num = _DAY_ID_MAP.get(did)
            if day_num is None:
                continue
            day = next((d for d in DAYS if d['id'] == day_num), None)
            if day is None:
                continue

            existing = next((e for e in day['exercises'] if e['name'] == name), None)
            if existing is not None:
                existing['type'] = ex['type']
                if 'default' in ex:
                    existing['default'] = ex['default']
                elif ex['type'] in ('bodyweight', 'excluded') and 'default' in existing:
                    del existing['default']
                existing['set_type'] = ex.get('set_type', 'standard')
                existing['no_amrap'] = ex.get('no_amrap', False)
                existing['variants'] = ex.get('variants', [])
            else:
                new_entry: dict = {
                    'name':     name,
                    'type':     ex['type'],
                    'set_type': ex.get('set_type', 'standard'),
                    'no_amrap': ex.get('no_amrap', False),
                    'variants': ex.get('variants', []),
                }
                if 'default' in ex:
                    new_entry['default'] = ex['default']
                day['exercises'].append(new_entry)


reload_exercises()
