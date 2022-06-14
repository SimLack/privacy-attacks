"""
with open ("amherst_558_0.25_nw_dynamic","r") as f:
    for num in range(3):
        while True:
            line = f.readline()
            if not line:
                print("return", num)
            print("line vor strip:")
            print(line)
            line = line.strip()
            print(line)
            if len(line) == 0:
                continue
            u, m = [int(i) for i in line.split()]
            print("u,m")
            print(u)
            print(m)
            break

        for i in range(m):
            line = f.readline()
            line = line.strip()
            u, v = [int(i) for i in line.split()]
            G.add_edge(u, v, weight = 1)
            GetNext.dict_add(G.node[u], 'out_degree', 1)
            GetNext.dict_add(G.node[v], 'in_degree', 1)
            GetNext.dict_add(G.graph, 'degree', 1)
            if not self.is_directed and u != v:
                G.add_edge(v, u, weight = 1)
                GetNext.dict_add(G.node[v], 'out_degree', 1)
                GetNext.dict_add(G.node[u], 'in_degree', 1)
                G.graph['degree'] += 1
"""
with open ("amherst_558_0.25_nw_dynamic","r") as f:
    line = f.readline()
    print(line)
    line = line.strip()
    u, m = [int(i) for i in line.split()]
