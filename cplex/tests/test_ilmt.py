# ---------------------------------------------------------------------------
# Licensed Materials - Property of IBM
# 5725-A06 5725-A29 5724-Y48 5724-Y49 5724-Y54 5724-Y55 5655-Y21
# Copyright IBM Corporation 2013, 2026. All Rights Reserved.
#
# US Government Users Restricted Rights - Use, duplication or
# disclosure restricted by GSA ADP Schedule Contract with
# IBM Corp.
# ---------------------------------------------------------------------------
"""
Tests ILMT logging.

No command line arguments are required.
"""
import os
import unittest
import sqlite3
import cplex
from cplextestcase import CplexTestCase

ILMT_DB_FILE_NAME='test.db'
# 60 (sec/min)* 60 (min/hour) * 24 (hour/day) = 86,400 sec/day
ILMT_TIME_LAPSE='86400'

# Temporarily set ILMT environment variables for testing.
os.environ['CPLEX_STUDIO_ILMT_DB_FILE_NAME'] = ILMT_DB_FILE_NAME
os.environ['CPLEX_STUDIO_ILMT_TIME_LAPSE'] = ILMT_TIME_LAPSE

# TODO CDO Move to integration tests ?
@unittest.skipIf(True, reason="Requires ILMT values")
class IlmtTest(CplexTestCase):

    def setUp(self):
        self._failSafeDelete(ILMT_DB_FILE_NAME)
        self.assertFalse(os.path.exists(ILMT_DB_FILE_NAME))

    def testSimple(self):
        with cplex.Cplex() as cpx:
            pass
        self.checkdb_simple()

    def testSequential(self):
        for i in range(10):
            with cplex.Cplex() as cpx:
                pass
        self.checkdb_simple()

    def testParallel(self):
        lst = []
        for i in range(10):
            lst.append(self._newCplex())
        for i in range(10):
            cpx = lst.pop()
            cpx.end()
        self.checkdb_simple()

    @staticmethod
    def get_record_count(cursor, table):
        sql = 'SELECT COUNT(*) FROM {0}'.format(table)
        cursor.execute(sql)
        return cursor.fetchone()

    @staticmethod
    def get_row(cursor, sql):
        cursor.execute(sql)
        return cursor.fetchone()

    def check_db_info(self, cur):
        row = self.get_record_count(cur, 'db_info')
        self.assertEqual(row[0], 1)
        row = self.get_row(cur, 'SELECT * FROM db_info')
        self.assertEqual(row['name'], 'ilmt')
        self.assertEqual(row['version'], 1)
        self.assertEqual(row['sqlite_version'], '3.33.0')

    def check_ilmt(self, cur):
        row = self.get_record_count(cur, 'ilmt')
        self.assertEqual(row[0], 1)
        row = self.get_row(cur, 'SELECT * FROM ilmt')
        self.assertEqual(row['id'], 1)
        self.assertTrue(row['pid'] > 0)
        self.assertTrue(row['start'] > 0)
        self.assertTrue(row['stop'] > 0)
        self.assertTrue(row['start'] <= row['stop'])
        self.assertEqual(row['count'], 0)

    def check_license_metric(self, cur):
        row = self.get_record_count(cur, 'license_metric')
        self.assertEqual(row[0], 2)
        row = self.get_row(cur,
                           'SELECT * FROM license_metric '
                           'WHERE id = 1')
        self.assertEqual(row['id'], 1)
        self.assertEqual(row['name'], 'AUTHORIZED_USER')

    def checkdb_simple(self):
        self.assertTrue(os.path.isfile(ILMT_DB_FILE_NAME))
        conn = None
        cur = None
        try:
            conn = sqlite3.connect(ILMT_DB_FILE_NAME)
            conn.row_factory = sqlite3.Row
            cur = conn.cursor()
            self.check_db_info(cur)
            self.check_license_metric(cur)
            self.check_ilmt(cur)
        finally:
            if cur is not None:
                cur.close()
            if conn is not None:
                conn.close()


def main():
    unittest.main()

if __name__ == '__main__':
    main()
