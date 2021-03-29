var num = 0
var state = getParamKey('z') == undefined ? 'none':'block';//初始状态
//隐藏功能
document.onkeydown = keyDownHandler;
function keyDownHandler(evt){
    evt = evt ? evt : window.event;
    if(evt.keyCode == 90){
        otherUpdateState(2);
        if(typeof(zfunc)=='function') zfunc();
    }
    if(typeof(bfunc)=='function' && (evt.keyCode == 66)) bfunc();
}
function body_obload(){
    if (state == 'block'){
        initState()
    }
    console.log(state)
}
//改状态
function initState(){
    var obj = document.getElementsByTagName("body")[0].getElementsByTagName("*");
    for(var i = 0;i < obj.length; i++){
        if(obj[i].id == 'z1'){
            console.log(state,'@@@')
            obj[i].style.display = state;
        }else if(obj[i].id == 'z2'){
            obj[i].style.display = state == 'none' ? 'block':'none';
        }
    }
}
//
function click_hidden_func(event,name){

    if(event.target.id=='z3')
    {
        var obj = event.currentTarget.getElementsByTagName("*");
        for(var i = 0;i < obj.length; i++){
            state = obj[i].style.display
            if(obj[i].id == 'z4'){
                obj[i].style.display = state == 'none' ? 'block':'none';
                (obj[i]).parentNode.style.opacity = state ? 1 : 1;
            }
        }
    }
}
//切换
function otherUpdateState(v=1){
    num += v
    state = (num > 1 && state == 'none') ? 'block':'none';
    console.log(state)
    initState();
}
//获取参数
function getParamKey(key){
    var str = window.location.search;
    if (str.indexOf(key) != -1) {
        var pos_start = str.indexOf(key) + key.length + 1;
        var pos_end = str.indexOf("&", pos_start);
        if (pos_end == -1) {
            return str.substring(pos_start)
        }
    }
}
//github CDN链接文件：javascript:goto('/')
cdndomain = "https://cdn.jsdelivr.net/gh/zbmain/note@master/"
function goto(filepath){
    var url = document.location.toString();
    var arr = url.split("//");
    url = arr[1].substring(arr[1].indexOf("/"));
    var arr = url.split("/");
    url = arr[1].substring(arr[0].indexOf("/"));
    url = cdndomain + url + filepath
    //url = url.replace(/^(http|https):\/\/[^/]+/, "");
    window.location = url
}
var _hmt = _hmt || [];
(function() {
  var hm = document.createElement("script");
  hm.src = "https://hm.baidu.com/hm.js?45ee8ee0b2719b61f8ff1c2b6cd306fe";
  var s = document.getElementsByTagName("script")[0]; 
  s.parentNode.insertBefore(hm, s);
})();