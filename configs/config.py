'''
Save all config setting
'''

class DBconfig:
    _db_config = {
        "host": "localhost",
        "user": "root",
        "password": "07032001",
        "database": "fashionstorewebsite",
    }

    @property
    def db_config(self):
        return self._db_config
