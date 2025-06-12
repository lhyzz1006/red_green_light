# 判断当前帧红灯是否亮
def is_red_light_on(frame):
    return True

def is_inside_red_area(x, y, red_mask):
    height, width = red_mask.shape
    if 0 <= x < width and 0 <= y < height:
        return red_mask[y, x] > 0
    return False