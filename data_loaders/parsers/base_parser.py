"""Base interface for converting annotations into project data records."""

from abc import ABC, abstractmethod


class BaseParser(ABC):
    def __init__(self, ann_file):
        self.ann_file = ann_file

    @abstractmethod
    def load_data(self):
        """Parse annotations and return one data record per retained image."""
