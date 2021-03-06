import numpy as np


def assemble(members, dofs):
    n_dof = np.max(dofs) + 1
    K = np.zeros((n_dof, n_dof))
    F = np.zeros(n_dof)

    for member_id, member in enumerate(members):
        nodes = member.nodes
        member_dofs = []
        for node in nodes:
            member_dofs.append(dofs[node])
        member_dofs = np.array(member_dofs).flatten()
        k_elem, f_elem_body = member.integrate()
        for n, i in enumerate(member_dofs):
            for m, j in enumerate(member_dofs):
                K[i, j] += k_elem[n, m]
            F[i] = f_elem_body[n]
    return K, F
