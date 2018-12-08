
class Model:
	def build_layer(self):
		pass

	def loss(self):
		pass


class DataLoader(object):
	def __init__(self):
		self.train_frames = []
		self.train_gts = []
		self.num_train = 0
		self.test_frames = []
		self.test_gts = []
		self.num_test = 0
		self.intrinsics = dict()

	def collect_frames(self):
		raise NotImplementedError()

	def get_train_example_with_idx(self, tgt_idx):
		raise NotImplementedError()

	def get_test_example_with_idx(self, tgt_idx):
		raise NotImplementedError()

	def is_valid_sample(self, frames, tgt_idx):
		raise NotImplementedError()
