import numpy as np

np.set_printoptions(precision=3, linewidth=120)


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
        k_elem = member.integral(member.stiffness)
        f_elem_body = member.integral(member.f_element_body)
        for n, i in enumerate(member_dofs):
            for m, j in enumerate(member_dofs):
                K[i, j] += k_elem[n, m]
            F[i] = f_elem_body[n]
    return K, F
