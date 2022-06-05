class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y
    
    def __eq__(self, other):
        if isinstance(other, Point):
            return self.x == other.x and self.y == other.y
        return False
    
    def __str__(self):
        return f'{(self.x, self.y)}'
    
    def __repr__(self):
        return f'{(self.x, self.y)}'
