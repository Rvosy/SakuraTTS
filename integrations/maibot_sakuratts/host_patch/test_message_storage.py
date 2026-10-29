"""Run in an isolated MaiBot container with /source-config mounted read-only."""

import asyncio
from contextlib import contextmanager
from pathlib import Path
import shutil
import sqlite3
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

shutil.copytree('/source-config', '/MaiMBot/config', dirs_exist_ok=True)

from sqlalchemy.exc import OperationalError
from sqlmodel import Field, Session, SQLModel, create_engine, select
from src.common.database import database
from src.common.utils.utils_message import MessageUtils
from src.services.send_service import _store_sent_message


class StoredVoice(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    platform_message_id: str
    text: str


class MessageStorageTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / 'messages.db'
        self.engine = create_engine('sqlite:///' + str(self.path),
                                    connect_args={'check_same_thread': False})
        self.addCleanup(self.engine.dispose)
        SQLModel.metadata.create_all(self.engine, tables=[StoredVoice.__table__])
        self.attempts = 0

        @contextmanager
        def session_scope():
            with Session(self.engine) as session:
                try:
                    yield session
                    session.commit()
                except Exception:
                    session.rollback()
                    raise

        self.scope = patch.object(database, 'get_db_session', session_scope)
        self.scope.start()
        self.addCleanup(self.scope.stop)
        self.frequency = patch.object(MessageUtils, 'fill_reply_frequency_if_available')
        self.frequency.start()
        self.addCleanup(self.frequency.stop)

        def record():
            self.attempts += 1
            return StoredVoice(platform_message_id='delivered-voice-1', text='こんにちは。')

        self.message = SimpleNamespace(raw_message=SimpleNamespace(components=[]), to_db_instance=record)

    def hold_writer(self):
        connection = sqlite3.connect(self.path, check_same_thread=False)
        connection.execute('BEGIN IMMEDIATE')
        return connection

    async def test_delivered_voice_survives_transient_writer_without_blocking_loop(self):
        connection = self.hold_writer()
        release = threading.Timer(1.25, connection.commit)
        release.start()
        try:
            task = asyncio.create_task(_store_sent_message(self.message))
            ticks = 0
            while not task.done():
                ticks += 1
                await asyncio.sleep(0.02)
            await task
        finally:
            release.join()
            connection.close()
        self.assertGreater(ticks, 10)
        self.assertGreater(self.attempts, 1)
        with Session(self.engine) as session:
            records = session.exec(select(StoredVoice)).all()
        self.assertEqual([(row.platform_message_id, row.text) for row in records],
                         [('delivered-voice-1', 'こんにちは。')])

    async def test_persistent_lock_remains_an_error_and_does_not_save_partial_record(self):
        connection = self.hold_writer()
        started = time.monotonic()
        try:
            with self.assertRaises(OperationalError) as error:
                await _store_sent_message(self.message)
            self.assertEqual(error.exception.orig.sqlite_errorcode & 0xff, sqlite3.SQLITE_BUSY)
        finally:
            connection.rollback()
            connection.close()
        self.assertLess(time.monotonic() - started, 8)
        with Session(self.engine) as session:
            self.assertEqual(session.exec(select(StoredVoice)).all(), [])

    async def test_non_busy_database_error_is_not_retried(self):
        StoredVoice.__table__.drop(self.engine)
        with self.assertRaises(OperationalError) as error:
            await _store_sent_message(self.message)
        self.assertEqual(error.exception.orig.sqlite_errorcode, sqlite3.SQLITE_ERROR)
        self.assertEqual(self.attempts, 1)


if __name__ == '__main__':
    unittest.main(verbosity=2)
