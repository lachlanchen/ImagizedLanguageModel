"""Lossless image-writing history with a visible, nonlinguistic cursor.

White output tiles still advance the writing state. Neither construction nor
reconstruction needs a font, alphabet, character ID or original text.
"""
import numpy as np


def _tiles(value):
    value=np.asarray(value)
    if value.dtype!=np.uint8 or value.ndim!=3 or value.shape[1:]!=(32,32):
        raise ValueError('actual uint8 writing images [N,32,32] required')
    return value


def _marker():
    value=np.full((4,32),255,dtype=np.uint8)
    value[:,8:24]=0
    return value


class WritingCanvas:
    """Immutable ordered writing pixels; cursor occupies a separate gutter."""
    def __init__(self,tiles,*,columns=32):
        if not isinstance(columns,int) or not 1<=columns<=128:
            raise ValueError('a physical line width of 1..128 image tiles is required')
        self._tiles=_tiles(tiles).copy();self._tiles.setflags(write=False)
        self.columns=columns

    @classmethod
    def empty(cls,*,columns=32):
        return cls(np.empty((0,32,32),dtype=np.uint8),columns=columns)

    @property
    def tiles(self):
        value=self._tiles.view();value.setflags(write=False)
        return value

    def __len__(self):return len(self._tiles)

    def append(self,tile):
        value=np.asarray(tile)
        if value.dtype!=np.uint8 or value.shape!=(32,32):
            raise ValueError('one actual emitted uint8 32-by-32 image required')
        return WritingCanvas(np.concatenate((self._tiles,value[None]),0),columns=self.columns)

    def page(self):
        # A cursor always has its own next cell, including at a line boundary.
        rows=len(self)//self.columns+1
        blocks=np.full((rows,self.columns,36,32),255,dtype=np.uint8)
        for i,tile in enumerate(self._tiles):blocks[i//self.columns,i%self.columns,:32]=tile
        row,column=divmod(len(self),self.columns);blocks[row,column,32:]=_marker()
        physical=blocks.transpose(0,2,1,3).reshape(rows*36,self.columns*32)
        return np.pad(physical,((0,(-len(physical))%32),(0,0)),constant_values=255)

    @classmethod
    def from_page(cls,page):
        value=np.asarray(page)
        if value.dtype!=np.uint8 or value.ndim!=2 or value.shape[0]%32 or value.shape[1]%32 or min(value.shape)<32:
            raise ValueError('a complete cursor page padded to 32-pixel dimensions is required')
        rows,columns=value.shape[0]//36,value.shape[1]//32
        if rows<1 or value.shape[0]!=32*((rows*36+31)//32) or not np.all(value[rows*36:]==255):
            raise ValueError('unknown row layout or nonwhite observer padding')
        blocks=value[:rows*36].reshape(rows,36,columns,32).transpose(0,2,1,3).reshape(-1,36,32)
        cursors=[]
        for i,block in enumerate(blocks):
            if np.array_equal(block[32:],_marker()):cursors.append(i)
            elif not np.all(block[32:]==255):raise ValueError('unknown mark in reserved cursor gutter')
        if len(cursors)!=1:raise ValueError('exactly one visible writing cursor is required')
        n=cursors[0]
        if rows!=n//columns+1 or not np.all(blocks[n:,:32]==255):
            raise ValueError('noncanonical rows or hidden writing beyond the cursor')
        return cls(blocks[:n,:32],columns=columns)
