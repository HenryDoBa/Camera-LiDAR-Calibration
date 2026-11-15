from calib.io_utils import list_files, pair_rows

def test_numeric_frame_order(tmp_path):
    for fn in [1, 20, 9, 11]:
        (tmp_path / f'test_fn{fn}.xyz').write_text('1 2 3\n')
    assert [x.stem for x in list_files(tmp_path, {'.xyz'})] == [
        'test_fn1','test_fn9','test_fn11','test_fn20']

def test_pair_frame_id_not_index_or_timestamp():
    lidars = [{'stem':f'test_fn{n}', 'status':'ok'} for n in (1,9,20)]
    cameras = [{'stem':f'Dev0_Image_w1920_h1200_fn{n}', 'status':'ok'} for n in (20,1,9)]
    matches = pair_rows(lidars,cameras,mode='frameid')
    assert [(a['stem'],b['stem']) for a,b,_ in matches] == [
        ('test_fn1','Dev0_Image_w1920_h1200_fn1'),
        ('test_fn9','Dev0_Image_w1920_h1200_fn9'),
        ('test_fn20','Dev0_Image_w1920_h1200_fn20')]
