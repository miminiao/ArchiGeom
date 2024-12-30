
# %% 线段合并测试
if 0 and __name__ == "__main__":
    import json
    import matplotlib.pyplot as plt
    const=Constant.default()
    with open("test/line_set/case_1.json",'r',encoding="utf8") as f:
        j_obj=json.load(f)
    lines=[]
    for obj in j_obj:
        if obj["object_name"]!="line":continue
        x1,y1,_=obj["start_point"]
        x2,y2,_=obj["end_point"]
        s,e=Node(x1,y1),Node(x2,y2)        
        if s.equals(e):continue
        lines.append(Edge(s,e))

    print(f"{len(lines)} lines before")
    merged_lines=MergeLineAlgo(lines,preserve_intersections=False).get_result()
    print(f"{len(merged_lines)} lines after")
    
    # for line in merged_lines:
    #     plt.plot([line.s.x,line.e.x],[line.s.y,line.e.y])
    # ax = plt.gca()
    # ax.set_aspect(1)
    # plt.show()

# %% 线段打断测试
if 0 and __name__ == "__main__":
    import json
    const=Constant.default()
    with open("./test/line_set/case_1.json",'r',encoding="utf8") as f:
        j_obj=json.load(f)
    print(f"{len(j_obj)} lines before")
    lines=[]
    for obj in j_obj:
        x1,y1,_=obj["start_point"]
        x2,y2,_=obj["end_point"]
        s,e=Node(x1,y1),Node(x2,y2)        
        if s.equals(e):continue
        lines.append(Edge(s,e))
    broken_lines=BreakEdgeAlgo([lines]).get_result()[0]
    print(f"{len(broken_lines)} lines after")

# %% 外轮廓测试
if 0 and __name__ == "__main__":
    import json
    const=Constant.default()
    with open("./test/find_outline/case_3.json",'r',encoding="utf8") as f:
        j_obj=json.load(f)
    edges:list[Edge]=[]
    for ent in j_obj:
        x1,y1,_=ent["start_point"]
        x2,y2,_=ent["end_point"]
        s=Node(x1,y1)
        e=Node(x2,y2)
        if s.equals(e):continue
        edges.append(Edge(s,e))
    outline=FindOutlineAlgo(edges).get_result()
    print(len(outline.edges),outline.area)
    plt.plot(*outline.xy)
    ax = plt.gca()
    ax.set_aspect(1)
    plt.show()

# %% 连通图测试
if 0 and __name__ == "__main__":
    import json
    import matplotlib.pyplot as plt
    from matplotlib.colors import TABLEAU_COLORS
    const=Constant.default()
    with open("./test/find_wall/case_13.json",'r',encoding="utf8") as f:
        j_obj=json.load(f)
    edges:list[Edge]=[]
    for ent in j_obj:
        if ent["object_name"]=="line" and ent["layer"]=="WALL":
            x1,y1,z1=ent["start_point"]
            x2,y2,z2=ent["end_point"]
            s=Node(x1,y1)
            e=Node(x2,y2)
            if s.equals(e):continue
            edges.append(Edge(s,e))
    edges=BreakEdgeAlgo([edges]).get_result()[0]
    con_graph=FindConnectedGraphAlgo(edges).get_result()
    print(len(con_graph))
    colors=list(TABLEAU_COLORS)
    for idx,g in enumerate(con_graph):
        color=colors[idx % len(colors)]
        for line in g:
            plt.plot(*line.to_array().T,color=color)
    ax = plt.gca()
    ax.set_aspect(1)
    plt.show()

# %% 连通图+外轮廓测试
if 0 and __name__ == "__main__":
    import json
    const=Constant.default()
    with open("./test/find_wall/case_1.json",'r',encoding="utf8") as f:
        j_obj=json.load(f)
    edges:list[Edge]=[]
    for ent in j_obj:
        if ent["object_name"]=="line" and ent["layer"]=="WALL":
            x1,y1,z1=ent["start_point"]
            x2,y2,z2=ent["end_point"]
            s=Node(x1,y1)
            e=Node(x2,y2)
            if s.equals(e):continue
            edges.append(Edge(s,e))
    edges=BreakEdgeAlgo([edges]).get_result()[0]
    con_graph=FindConnectedGraphAlgo(edges).get_result()
    outlines=[FindOutlineAlgo(edges).get_result() for edges in con_graph]
    print(len(outlines))
    import matplotlib.pyplot as plt
    from matplotlib.colors import TABLEAU_COLORS
    colors=list(TABLEAU_COLORS)
    for idx,g in enumerate(outlines):
        color=colors[idx % len(colors)]
        for line in g.edges:
            plt.plot(*line.to_array().T,color=color)
    ax = plt.gca()
    ax.set_aspect(1)
    plt.show()

# %% 线段合并测试，带优先级比较
if 0 and __name__ == "__main__":
    import json,random
    import matplotlib.pyplot as plt
    const=Constant.default()

    # with open("./test/merge_line/case_1.json",'r',encoding="utf8") as f:
    #     j_obj=json.load(f)
    # edges:list[Edge]=[]
    # for ent in j_obj:
    #     if ent["object_name"]=="line" and ent["layer"]=="WALL":
    #         x1,y1,z1=ent["start_point"]
    #         x2,y2,z2=ent["end_point"]
    #         s=Node(x1,y1)
    #         e=Node(x2,y2)
    #         if s.equals(e):continue
    #         edges.append(Edge(s,e))

    lines:list[LineSeg]=[]
    limits=(0,10000,1000)
    random.seed(0)
    for i in range(10):
        s=Node(random.random()*(limits[1]-limits[0])+limits[0],0)
        # e=Node(random.random()*(limits[1]-limits[0])+limits[0],0)
        e=Node(s.x+1000,0)
        lw=random.random()*limits[2]
        # s=Node(random.randint(limits[0],limits[1]),0)
        # e=Node(random.randint(limits[0],limits[1]),0)
        # lw=random.randint(0,limits[2])
        lines.append(LineSeg(s,e))
        lines[-1].lw,lines[-1].rw=lw,0

    plt.subplot(2,1,1)
    for i,line in enumerate(lines):
        plt.plot([line.s.x,line.e.x],[line.s.y+line.lw+line.rw,line.e.y+line.lw+line.rw])

    print(f"{len(lines)} lines before")
    def compare(self,a:Edge,b:Edge): 
        if a is None: return -1
        if b is None: return 1
        if abs(a.lw+a.rw-(b.lw+b.rw))<const.TOL_DIST: return 0
        elif a.lw+a.rw>b.lw+b.rw: return 1
        else: return -1
    merged_lines=MergeLineAlgo(lines,preserve_intersections=False,compare=compare).get_result()
    print(f"{len(merged_lines)} lines after")

    plt.subplot(2,1,2)
    for i,line in enumerate(merged_lines):
        plt.plot([line.s.x,line.e.x],[line.s.y+line.lw+line.rw,line.e.y+line.lw+line.rw])

    plt.show()

    # CASE_ID="6"

    # with open(f"./test/merge_line/case_{CASE_ID}.json",'w',encoding="utf8") as f:
    #     json.dump(lines,f,ensure_ascii=False,default=lambda x:x.__dict__)
    # with open(f"./test/merge_line/case_{CASE_ID}_out.json",'w',encoding="utf8") as f:
    #     json.dump(merged_lines,f,ensure_ascii=False,default=lambda x:x.__dict__)

# %% 合并相交环测试
if 0 and __name__ == "__main__":
    import json
    import matplotlib.pyplot as plt
    from matplotlib.colors import TABLEAU_COLORS
    from tool.converter.json_converter import cad_polyline_to_loop
    colors=list(TABLEAU_COLORS)
    const=Constant.default()
    # const=Constant("split_loop",tol_area=1e3,tol_dist=1e-2)

    CASE_ID = "12.2"  ################ TEST #################

    with open(f"test/split_loop/case_{CASE_ID}.json",'r',encoding="utf8") as f:
        j_obj=json.load(f)
    loops=cad_polyline_to_loop(j_obj)

    with Timer(tag="split_loop"):
        split_loops=SplitIntersectedLoopsAlgo(loops,False,False,const=const).get_result()
    split_loops.sort(key=lambda loop:loop.area)

    print(len(split_loops))
    _draw_loops(split_loops,show_node=False,show_text=False,show=True)

    # 输出标准结果
    # with open(f"test\split_loop\case_{CASE_ID}_out.json",'w',encoding="utf8") as f:
    #     json.dump([loop.area for loop in split_loops],f,ensure_ascii=False)

# %% 找回环测试
if 1 and __name__ == "__main__":
    import json
    import matplotlib.pyplot as plt
    from matplotlib.colors import TABLEAU_COLORS
    const=Constant.default()
    with open("./test/find_loop/case_1.json",'r',encoding="utf8") as f:
        j_obj=json.load(f)
    edges:list[Edge]=[]
    for ent in j_obj:
        if ent["object_name"]=="line":
            x1,y1,z1=ent["start_point"]
            x2,y2,z2=ent["end_point"]
            s=Node(x1,y1)
            e=Node(x2,y2)
            if s.equals(e):continue
            edges.append(Edge(s,e))
    edges=BreakEdgeAlgo([edges]).get_result()[0]
    con_graph=FindConnectedGraphAlgo(edges).get_result()
    print(len(con_graph))
    colors=list(TABLEAU_COLORS)
    for idx,g in enumerate(con_graph):
        color=colors[idx % len(colors)]
        for line in g:
            plt.plot(*line.to_array().T,color=color)
    ax = plt.gca()
    ax.set_aspect(1)
    plt.show()

