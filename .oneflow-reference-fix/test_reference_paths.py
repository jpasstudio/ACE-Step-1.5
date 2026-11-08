import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from oneflow_reference_paths import migrate_uploaded_reference, LegacyVoiceReferenceMiddleware

class ReferencePathsTest(unittest.TestCase):
 def setUp(self):
  self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name);self.legacy=self.root/'legacy';self.pool=self.root/'pool';self.legacy.mkdir();self.pool.mkdir();self.source=self.legacy/('a'*32+'.wav');self.source.write_bytes(b'x'*2048)
 def tearDown(self):self.temp.cleanup()
 def test_known_upload_migrates_without_changing_original(self):
  target=Path(migrate_uploaded_reference(str(self.source),self.legacy,self.pool));self.assertEqual(target.read_bytes(),self.source.read_bytes());self.assertTrue(self.source.exists());self.assertEqual(migrate_uploaded_reference(str(self.source),self.legacy,self.pool),str(target))
 def test_unknown_absolute_path_is_not_rewritten(self):self.assertEqual(migrate_uploaded_reference('/etc/passwd',self.legacy,self.pool),'/etc/passwd')
 def test_traversal_symlink_and_unrecognized_names_rejected(self):
  for value in [str(self.legacy/'../secret.wav'),str(self.legacy/'secret.wav')]:
   with self.assertRaises(ValueError):migrate_uploaded_reference(value,self.legacy,self.pool)
  link=self.legacy/('b'*32+'.wav');link.symlink_to(self.source)
  with self.assertRaises(ValueError):migrate_uploaded_reference(str(link),self.legacy,self.pool)
 def test_asgi_only_rewrites_reference_and_preserves_the_approved_payload(self):
  seen={};body=json.dumps({'reference_audio_path':str(self.source),'lyrics':'approved Thai lyrics','seed':17}).encode()
  async def downstream(scope,receive,send):seen.update(json.loads((await receive())['body']))
  middleware=LegacyVoiceReferenceMiddleware(downstream,self.legacy,self.pool)
  async def receive():return {'type':'http.request','body':body,'more_body':False}
  async def send(message):pass
  asyncio.run(middleware({'type':'http','method':'POST','path':'/release_task','headers':[(b'content-type',b'application/json')]},receive,send))
  self.assertEqual(seen['lyrics'],'approved Thai lyrics');self.assertEqual(seen['seed'],17);self.assertEqual(Path(seen['reference_audio_path']).parent,self.pool)

if __name__=='__main__':unittest.main()
