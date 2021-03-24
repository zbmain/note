# 2.5 规则过滤器

## 学习目标

- 目标
  - 了解规则过滤器的作用
- 应用
  - 无

### 2.5.1 规则过滤器

- 什么是规则过滤器: **为了保证推荐内容的多样性, 合理性, 每次的推荐内容会在内部和上次数据间做一些比较和过滤操作**
- 规则过滤器作用：防止推荐数据重复, 并可按照指定规则选择性推荐，过滤掉不同用户发表的相同的帖子(帖子ID不一样，内容相似或者相同)
  - 如何比较推荐出去的两个pid内容相同？

规则过滤器主体函数代码：

```python
import requests
import cv2

def rfilter(r_data):
    with _driver.session() as session:
        def get_url(pid):
            cypher = "match(a:SuperfansPost{pid:%d}) return a.iv_url" % (pid)
            record = session.run(cypher)
            result = list(map(lambda x: x[0], record))
            if result:
                return result[0]
        # 获取所有要推荐的图片url帖子地址
        url_list = list(
            filter(
                lambda x: x is not None, map(
                    lambda x: get_url(x), r_data)))
		# 1、帖子PID与图片地址绑定
    url_dict = dict(zip(url_list, r_data))
    # 2、去重函数
    url_list = d_hash_serve(url_list)
    # 3、将去重之后的url和PID取出结果传入排序模块
    result = list(map(lambda x: url_dict[x], url_list))
    return result
```

那如何进行去重，分析步骤如下：

* 1、下载所有图片到本地，保留图片路径列表
* 2、循环所有图片列表，两两之间进行比较
  * 通过opencv将两个图片的指纹进行汉明距离比较

比较图片相同方式，获取帖子的图片地址然后进行相似度计算：

```python
def d_hash_serve(url_list):
    """去重函数逻辑
    """
    # 下载所有图片
    default_image = "test.png"

    def download(url):
        path = "./recomm/img_compare/"
        try:
            img = requests.get(url)
            with open(path + url + ".jpg", "wb") as fp:
                fp.write(img.text)
            return path + url + ".jpg"
        except BaseException:
            return path + default_image

    # 1、得到下载到本地之后的图片列表
    url_path = list(map(lambda url: download(url), url_list))

    # 2、得到本地路径与CDN路径的对应字典
    url_dict = dict(zip(url_path, url_list))

    # 3、compare_img函数对该目录下的图片文件进行比较
    compare_list = compare_img(path)

    # 4、获得比较之后应该删除的重复图片CDN路径
    compared_delete_list = list(
        map(lambda x: url_dict[x], list(set(compare_list[::2]))))

    # 5、根据CDN路径进行列表元素删除
    list(map(lambda x: url_list.remove(x), compared_delete_list))

    return url_list
  
  
def compare_img(root_path):
    """
    比较图片 (Main)
    :param root_path: 目标文件夹
    """
    compared_list = []
    # 获取目标文件夹下所有图片路径集合
    img_list = get_all_img_list(root_path)
    # 遍历目标文件夹下所有图片进行两两比较
    for file1 in img_list:
        # 已经发现是相同的图片不再比较
        if file1 in compared_list:
            continue
        im1 = cv2.imdecode(
            np.fromfile(
                file1,
                dtype=np.uint8),
            cv2.IMREAD_UNCHANGED)
        print(im1)
        if im1 is None:
            continue
        im1_size = os.path.getsize(file1)
        # 获取图片指纹
        img_fingerprints1 = get_img_gray_bit(im1)
        print("第一张的指纹:", img_fingerprints1)
        for file2 in img_list:
            if file1 != file2:
                # im2 = cv2.imread(files2)
                im2 = cv2.imdecode(
                    np.fromfile(
                        file2,
                        dtype=np.uint8),
                    cv2.IMREAD_UNCHANGED)
                if im2 is None:
                    continue
                im2_size = os.path.getsize(file2)
                print(im2_size)
                # 如果两张图片字节数大小一样再判断汉明距离
                if im1_size != im2_size:
                    # 获取图片指纹
                    img_fingerprints2 = get_img_gray_bit(im2)
                    print("第二张的指纹:", img_fingerprints2)
                    compare_result = get_mh(
                        img_fingerprints1, img_fingerprints2)
                    print(compare_result)
                    # 汉明距离等于0，说明两张图片完全一样
                    if compare_result == 0:

                        compared_list.append(file2)
                        compared_list.append(file1)
    return compared_list

  
def get_all_img_list(root_path):
    """
    获取目标文件夹下所有图片路径集合
    :param root_path: 目标文件夹
    :return: 图片集合
    """
    img_list = []
    # 获取目标文件夹下所有元组
    root = os.walk(root_path)
    # 循环元组，获取目标文件夹下所有图片路径集合
    for objects in root:
        for obj in objects:
            if "/" in str(obj):
                # 记录文件夹路径
                path = str(obj)
            elif len(obj) > 0:
                # 如果是文件，判断是否是图片。如果是图片则保存进
                for file in obj:
                    if "." in str(file) and is_image_file(file) == 1:
                        full_path = path + "/" + str(file)
                        img_list.append(full_path.replace("\\", "/"))
    return img_list
  

def get_img_gray_bit(img, resize=(32, 32)):
    """
    获取图片指纹
    :param img: 图片
    :param resize: Resize的图片大小
    :return: 图片指纹
    """
    # 修改图片大小
    image_resize = cv2.resize(img, resize, interpolation=cv2.INTER_CUBIC)
    # 修改图片成灰度图
    image_gray = cv2.cvtColor(image_resize, cv2.COLOR_BGR2GRAY)
    # 转换灰度图成浮点型
    image_gray_f = np.float32(image_gray)
    # 获取灰度图的DCT集合
    image_gray_dct = cv2.dct(image_gray_f)
    # 获取灰度图DCT集合的左上角8*8
    # gray_dct_ul64_list = get_gray_dct_ul64_list(image_gray_dct)
    gray_dct_ul64_list = image_gray_dct[0:8, 0:8]
    # 获取灰度图DCT集合的左上角8*8对应的平均值
    # gray_dct_ul64_avg = get_gray_dct_ul64_avg(gray_dct_ul64_list)
    gray_dct_ul64_avg = cv2.mean(gray_dct_ul64_list)
    # 获取图片指纹
    img_fingerprints = get_img_fingerprints(
        gray_dct_ul64_list, gray_dct_ul64_avg)
    return img_fingerprints
  

def get_img_fingerprints(gray_dct_ul64_list, gray_dct_ul64_avg):
    """
    获取图片指纹：遍历灰度图左上8*8的所有像素，比平均值大则记录为1，否则记录为0。
    :param gray_dct_ul64_list: 灰度图左上8*8的所有像素
    :param gray_dct_ul64_avg: 灰度图左上8*8的所有像素平均值
    :return: 图片指纹
    """
    img_fingerprints = ''
    avg = gray_dct_ul64_avg[0]
    for i in range(8):
        for j in range(8):
            if gray_dct_ul64_list[i][j] > avg:
                img_fingerprints += '1'
            else:
                img_fingerprints += '0'
    return img_fingerprints


def get_mh(img_fingerprints1, img_fingerprints2):
    """
    获取汉明距离
    :param img_fingerprints1: 比较对象1的指纹
    :param img_fingerprints2: 比较对象2的指纹
    :return: 汉明距离
    """
    hm = 0
    for i in range(0, len(img_fingerprints1)):
        if img_fingerprints1[i] != img_fingerprints2[i]:
            hm += 1
    return hm


def is_image_file(file_name):
    """
    判断文件是否是图片
    :param file_name: 文件名称(包含后缀信息)
    :return: 1:图片，0:非图片
    """
    ext = (os.path.splitext(file_name)[1]).lower()
    if ext == ".jpg" or ext == ".jpeg" or ext == ".bmp" or ext == ".png":
        return 1
    return 0
```

在_get_recomm函数中加入规则过滤器

```python
def _get_recomm(IP, uid):
    """整体推荐流程
    :param IP: 用户的IP
    :param uid: 用户ID
    :return:
    """
    # 1、召回的模块接口实现
    hot_data = _get_hot()
    last_data = get_last()
    v_data = get_v()
    r_data = get_r(uid)
    random_data = get_random()
    # 合并召回结果
    all_data = [hot_data] + [last_data] + [v_data] + [r_data] + [random_data]
    # 金字塔结构实现，给PID分配权重排序输出pid
    j_data = pyramid_array(all_data)
    # 将数据写入金字塔缓存
    r_data = j_data_write(uid, j_data)
    # 规则过滤器
    f_data = rfilter(r_data)

    return r_data
```

### 2.5.2 总结

* 了解规则过滤器的作用和实现