def get_dependency_dict(on_immutable: bool, on_mutable: bool):
    # @TODO dependent on device state?
    return {"on_immutable": on_immutable, "on_mutable": on_mutable}