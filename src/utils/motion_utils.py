import torch
import torch.nn.functional as F

from utils.data_utils import FFT


def generate_list(size, n_movements, mingap=4, acs=24):
    # Generate a list of random numbers that sum to size with a minimum gap of mingap and if movemement is in the center of the size then discard and sample again

    slack = size - mingap * (n_movements - 1)

    steps = torch.randint(0, slack, (1,))[0]

    increments = torch.hstack(
        [
            torch.ones((steps,), dtype=torch.long),
            torch.zeros((n_movements,), dtype=torch.long),
        ]
    )

    increments = increments[torch.randperm(increments.shape[0])]

    locs = torch.argwhere(increments == 0).flatten()
    return torch.cumsum(increments, dim=0)[locs] + mingap * torch.arange(0, n_movements)


def get_rand_int(data_range, size=None):
    if size is None:
        rand = torch.randint(data_range[0], data_range[1], size=(1,))
        if rand == 0:
            rand = rand + 1
    else:
        rand = torch.randint(data_range[0], data_range[1], size=size)
    return rand


def extract_movements(indices):

    phase_lines = indices.shape[0]
    mask = torch.zeros((phase_lines, phase_lines), dtype=torch.long)
    count = 1
    for i in range(phase_lines):
        if i != phase_lines - 1 and indices[i] == 1 and indices[i + 1] == 1:
            mask[:, i] = count
        elif i != phase_lines - 1 and indices[i] == 1 and indices[i + 1] == 0:
            mask[:, i] = count
            count += 1
        elif i == phase_lines - 1 and indices[i] == 1:
            mask[:, i] = count
        else:
            pass

    return mask


def extract_movement_groups(motionline_indcies, make_list=False):
    """_summary_

    Returns:
        torch.Tensor: motion_groups (phase_encoding, frequency_encoding) where each pixel is assigned a group number

    example:
    motion_groups = extract_movement_groups(motion_mask.squeeze().sum(0) // phase_encoding)

    num_movements = (motion_groups).unique().numel() - 1
    """
    phase_lines = motionline_indcies.shape[0]

    motion_groups = torch.zeros(
        (phase_lines, phase_lines), dtype=torch.long, device=motionline_indcies.device
    )

    count = 1
    for phase_line in range(phase_lines):
        if (
            phase_line != phase_lines - 1
            and motionline_indcies[phase_line] == 1
            and motionline_indcies[phase_line + 1] == 1
        ):
            motion_groups[:, phase_line] = count
        elif (
            phase_line != phase_lines - 1
            and motionline_indcies[phase_line] == 1
            and motionline_indcies[phase_line + 1] == 0
        ):
            motion_groups[:, phase_line] = count
            count += 1
        elif phase_line == phase_lines - 1 and motionline_indcies[phase_line] == 1:
            motion_groups[:, phase_line] = count
        else:
            pass

    if make_list:
        unique = torch.unique(motion_groups).nonzero().squeeze()

        counts = unique.numel()

        motion_lists = torch.zeros(
            (counts, *motion_groups.shape),
            dtype=torch.long,
            device=motionline_indcies.device,
        )

        for i in range(counts):

            motion_lists[i, (motion_groups == i + 1).bool()] = 1
        motion_groups = motion_lists

    return motion_groups


def rotation_matrix_2d(angle, device=None):
    """2D rotation matrix."""
    angle = torch.deg2rad(angle)
    return torch.tensor(
        [[torch.cos(angle), -torch.sin(angle)], [torch.sin(angle), torch.cos(angle)]],
        device=device,
    )


def motion_simulation2D(
    image_2d,
    n_movements=None,
):

    ksp_corrupt = FFT(image_2d)
    x, num_lines = ksp_corrupt.shape

    channels = 1

    if n_movements is None:
        n_movements = get_rand_int([5, 20]).item()

    mingap = num_lines // n_movements

    acs = int(num_lines * 0.08)  # 8% of the lines are ACS lines

    rand_list = generate_list(num_lines, n_movements, mingap, acs)
    motion_rng = [1, n_movements]

    mask = torch.zeros((x, num_lines), dtype=torch.long)
    rotations = torch.zeros((n_movements,))
    translations = torch.zeros((n_movements, 2))
    shift_x_rng = [[-10, 10]] * motion_rng[1]
    shift_y_rng = [[-10, 10]] * motion_rng[1]
    rotate_rng = [[-10, 10]] * motion_rng[1]
    shift_w_rng = [[-2, 2]] * motion_rng[1]
    motion_list = [[]] * motion_rng[1]
    w_rng = [1, 10]

    for motion in range(n_movements):

        shift = [
            get_rand_int(shift_x_rng[motion]).item(),
            get_rand_int(shift_y_rng[motion]).item(),
        ]
        angle = get_rand_int(rotate_rng[motion])
        rotation = rotation_matrix_2d(angle)
        torch_affine = torch.tensor([[1, 0, shift[0]], [0, 1, shift[1]]]).float()
        torch_affine[:2, :2] = rotation
        torch_affine = torch_affine.view(1, 2, 3)

        torch_affine[:, :, -1] /= (torch.tensor(image_2d[0, ...].shape) * 2.0) - 1

        grid = F.affine_grid(
            torch_affine, (1, channels, x, num_lines), align_corners=True
        ).to(image_2d.device)
        image_2d_transformed_real = F.grid_sample(
            image_2d.unsqueeze(0).unsqueeze(0).real,
            grid.float(),
            mode="bilinear",
            padding_mode="border",
            align_corners=False,
        )

        image_2d_transformed_imag = F.grid_sample(
            image_2d.unsqueeze(0).unsqueeze(0).imag,
            grid.float(),
            mode="bilinear",
            padding_mode="border",
            align_corners=False,
        )

        image_2d_transformed = (
            image_2d_transformed_real + 1j * image_2d_transformed_imag
        )

        # ksp for shifted image
        ksp_shiftnrotate = FFT(image_2d_transformed).squeeze()

        # replace the ksp
        w_start = rand_list[motion]
        w_end = w_start + get_rand_int(w_rng)

        ksp_corrupt[..., w_start:w_end] = ksp_shiftnrotate[..., w_start:w_end]
        mask[:, w_start:w_end] = 1
        motion_list.append([w_start.item(), w_end])

        rotations[motion] = angle
        translations[motion, :] = torch.tensor(shift)

    return ksp_corrupt, mask, rotations, translations
