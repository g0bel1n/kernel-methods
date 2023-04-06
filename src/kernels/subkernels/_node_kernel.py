from ._base import NodeKernel

class BinaryNodeKernel(NodeKernel):

    def _kernel(self, sub1, sub2) -> int:
        return int(sub1==sub2)