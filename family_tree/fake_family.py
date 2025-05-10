import datetime
import random
import numpy as np
import pandas as pd
import networkx as nx


def first_name_list(list_source: str = "social_security"):
    if list_source == "social_security":
        return pd.read_csv(
            "../synthetic_data/first_names/first_names_social_security.csv"
        )
    else:
        raise ValueError("List source not available")


def non_english_first_name_list(list_language: str = "hebrew"):
    if list_language == "hebrew":
        return pd.read_csv(
            "../synthetic_data/non_english_first_names/hebrew_first_names.csv"
        )
    else:
        raise ValueError("List language not available")


def surname_list(list_source: str = "jewish_surnames"):
    if list_source == "jewish_surnames":
        return pd.read_csv("../synthetic_data/surnames/jewish_surnames.csv")
    else:
        raise ValueError("List source not available")


def generate_person(
    df_first_names: pd.DataFrame,
    df_surnames: pd.DataFrame,
    df_non_englsh_names: pd.DataFrame = None,
    birth_year: int = None,
    gender: str = None,
    sex: str = None,
):
    if gender is None:
        gender = random.sample(
            ["male", "female", "nonbinary", "other"], k=1, counts=[10, 10, 1, 1]
        )[0]
    elif gender not in ["male", "female", "nonbinary", "other"]:
        raise ValueError("gender must be one of male, female, nonbinary, or other")

    if sex is None:
        if gender == "male":
            sex = random.sample(["male", "female"], k=1, counts=[20, 1])[0]
        elif gender == "female":
            sex = random.sample(["female", "male"], k=1, counts=[20, 1])[0]
        else:
            sex = random.sample(["female", "male"], k=1, counts=[1, 1])[0]
    elif sex not in ["male", "female", "intersex"]:
        raise ValueError("sex must be one of male, female, or intersex")

    if birth_year is None:
        birth_year = random.sample(
            population=range(2025, 1880, -1),
            k=1,
            counts=[x for x in range(1, 2025 - 1880 + 1)],
        )[0]
    birth_datetime = (
        datetime.datetime.strptime(str(birth_year), "%Y")
        + datetime.timedelta(days=random.randint(0, 365))
    ).date()
    # formula based off of life expectancies from CDC
    # (https://healthdata.gov/dataset/NCHS-Death-rates-and-life-expectancy-at-birth/4r8i-dqgb/about_data)
    life_expectancy = int(521.64 * np.log(birth_year) - 3887)
    lifespan = (
        life_expectancy + int(random.normalvariate(0, 15))
    ) * 365 + random.randint(1, 365)
    death_datetime = birth_datetime + datetime.timedelta(days=lifespan)

    if gender in ["male", "female"]:
        df_first_names_small = df_first_names.loc[
            (df_first_names["year"] == birth_datetime.year)
            & (df_first_names["sex"] == gender[0])
        ]
    elif gender == "nonbinary":
        df_first_names_small = df_first_names.loc[
            (df_first_names["year"] == birth_datetime.year)
        ]
    else:
        df_first_names_small = df_first_names

    if df_non_englsh_names is not None:
        if gender in ["male", "female"]:
            df_non_englsh_names_small = df_non_englsh_names.loc[
                df_non_englsh_names["sex"] == gender[0]
            ]
        else:
            df_non_englsh_names_small = df_non_englsh_names

    names = random.sample(
        df_first_names_small["name"].tolist(),
        k=5,
        counts=df_first_names_small["count"].tolist(),
    )
    first_name = names[0]
    middle_names_count = random.sample(range(3), 1)[0]
    middle_name = " ".join(names[1 : 1 + middle_names_count])
    last_name = df_surnames["surname"].sample().iloc[0]
    if df_non_englsh_names is not None:
        non_englsh_name_list = random.sample(
            df_non_englsh_names_small["name"].tolist(),
            k=3,
            counts=df_non_englsh_names_small["first_name_count"].tolist(),
        )
        non_english_name = " ".join(non_englsh_name_list[0 : 1 + middle_names_count])
    else:
        non_english_name = None

    person_dict = {
        "first_name": first_name,
        "middle_name": middle_name,
        "last_name": last_name,
        "maiden_name": last_name,
        "preferred_name": first_name,
        "nickname": first_name,
        "non_english_name": non_english_name,
        "date_of_birth": birth_datetime,
        #         "date_of_adoption": adoption_date,
        "date_of_death": death_datetime,
        "gender": gender,
        "sex": sex,
    }
    return person_dict


def generate_relationship(
    relationship_type: str = None, date_start: str = None, date_end: str = None
):
    if relationship_type is None:
        relationship_type = random.sample(
            ["parent_child", "partnership", "owner_pet"],
            k=1,
            counts=[20, 20, 0],
        )[0]
    elif relationship_type not in ["parent_child", "partnership", "owner_pet"]:
        raise ValueError(
            "relationship_type must be one of parent_child, partnership, or owner_pet"
        )
    relationship_subtype = None
    start_datetime = None
    end_datetime = None
    if relationship_type == "partnership":
        relationship_subtype = random.sample(
            ["marriage", "partnered"], k=1, counts=[10, 1]
        )[0]
        if date_start is None:
            start_year = random.sample(
                population=range(2025, 1900, -1),
                k=1,
                counts=[x for x in range(1, 2025 - 1900 + 1)],
            )[0]
        start_datetime = (
            datetime.datetime.strptime(str(start_year), "%Y")
            + datetime.timedelta(days=random.randint(0, 365))
        ).date()

        duration = random.sample(population=range(5 * 365, 70 * 365), k=1)[0]

        end_datetime = start_datetime + datetime.timedelta(days=duration)

    relationship_dict = {
        "type": relationship_type,
        "subtype": relationship_subtype,
        "date_start": start_datetime,
        "date_end": end_datetime,
    }
    return relationship_dict


def find_partnerships(
    df_people: pd.DataFrame, df_relationships: pd.DataFrame, relationships_people: list
):
    for idx, val in df_relationships.loc[
        df_relationships["type"] == "partnership"
    ].iterrows():
        # Let's put bounds that they get married sometime between the age of 19 and 70
        starter_birth_date_upper = val.date_start - datetime.timedelta(days=19 * 365)
        starter_birth_date_lower = val.date_start - datetime.timedelta(days=70 * 365)
        eligible_starter = df_people.loc[
            (df_people["date_of_birth"] <= starter_birth_date_upper)
            & (df_people["date_of_birth"] >= starter_birth_date_lower)
        ]
        eligible_starter_ids = eligible_starter["id"].tolist()
        # remove people who are already in relationships
        ineligible_ids = []
        if len(relationships_people) > 0:
            df_ineligible = pd.DataFrame(relationships_people)
            df_ineligible = df_ineligible.merge(
                df_relationships, how="left", left_on="relationship_id", right_on="id"
            )
            ineligible_ids = df_ineligible.loc[
                (
                    (df_ineligible["date_start"] <= val.date_start)
                    & (df_ineligible["date_end"] >= val.date_start)
                )
                | (
                    (df_ineligible["date_start"] >= val.date_start)
                    & (df_ineligible["date_start"] <= val.date_end)
                ),
                "people_id",
            ].tolist()
        #             if len(ineligible_ids) > 0:
        #                 print(f"ineligible ids for relationship {val.id} are {ineligible_ids}")
        eligible_starter_ids = [
            x for x in eligible_starter_ids if x not in ineligible_ids
        ]
        random.shuffle(eligible_starter_ids)
        # Keep trying to find a set of working partners

        for partner_1_id in eligible_starter_ids:
            partner_1 = df_people.loc[df_people["id"] == partner_1_id].to_dict(
                orient="records"
            )[0]
            # Let's put bounds on the partner being born within 10 years
            birth_range_lower = partner_1["date_of_birth"] - datetime.timedelta(
                days=10 * 365
            )
            birth_range_upper = partner_1["date_of_birth"] + datetime.timedelta(
                days=10 * 365
            )
            eligible_partners = df_people.loc[
                (df_people["date_of_birth"] <= birth_range_upper)
                & (df_people["date_of_birth"] >= birth_range_lower)
                #                 & (df_people["date_of_death"] > val.date_end)
                & (df_people["id"] != partner_1_id)
                & (~df_people["id"].isin(ineligible_ids))
            ]

            if eligible_partners.shape[0] > 0:
                partner_2_id = random.sample(eligible_partners["id"].tolist(), k=1)[0]
                # partner_2 = df_people.loc[df_people["id"] == partner_2_id].to_dict(
                #     orient="records"
                # )[0]
                partner_ids = [partner_1_id, partner_2_id]

                # Update last names for both partners when partnership is formed
                # Use the first male/nonbinary gendered last name in the relationship
                last_name = (
                    df_people.loc[df_people["id"].isin(partner_ids)]
                    .sort_values(["gender"], ascending=False)["last_name"]
                    .iloc[0]
                )
                for partner_id in partner_ids:
                    df_people.loc[df_people["id"] == partner_id, "last_name"] = last_name
                    relationships_people += [
                        {
                            "relationship_id": val.id,
                            "people_id": partner_id,
                            "title": "partner",
                        }
                    ]
                break
    return relationships_people


def find_children(
    df_people: pd.DataFrame, df_relationships: pd.DataFrame, relationships_people: list
):
    df_relationships_people = pd.DataFrame(relationships_people)
    # Associate each parent_child relationship to a partnership
    partnership_ids = df_relationships.loc[
        df_relationships["type"] == "partnership", "id"
    ].tolist()
    parent_child_ids = df_relationships.loc[
        df_relationships["type"] == "parent_child", "id"
    ].tolist()
    partner_parent_zip = list(zip(parent_child_ids, partnership_ids))

    # Keep track of assigned children
    assigned_children = set()

    for ppz in partner_parent_zip:
        parent_child_id = ppz[0]
        partnership_id = ppz[1]

        # Associate parents to parent_child relationship
        parent_ids = df_relationships_people.loc[
            df_relationships_people["relationship_id"] == partnership_id, "people_id"
        ].tolist()

        partnership_start = df_relationships.loc[
            df_relationships["id"] == partnership_id, "date_start"
        ].iloc[0]
        partnership_end = df_relationships.loc[
            df_relationships["id"] == partnership_id, "date_end"
        ].iloc[0]
        possible_children_ids = df_people.loc[
            (~df_people["id"].isin(parent_ids))
            & (df_people["date_of_birth"] > partnership_start)
            & (df_people["date_of_birth"] < partnership_end)
            # Add filter to exclude already assigned children
            & (~df_people["id"].isin(assigned_children)),
            "id",
        ].tolist()
        if len(possible_children_ids) > 0:
            # Prevent sibling relationships
            for partnership in partnership_ids:
                id_list = df_relationships_people.loc[
                    df_relationships_people["relationship_id"] == partnership,
                    "people_id",
                ].tolist()
                if id_list[0] in possible_children_ids:
                    possible_children_ids = [
                        x
                        for x in possible_children_ids
                        if x not in random.sample(id_list, k=len(id_list) - 1)
                    ]
            # Increase the probabilty of one of the children belonging to a partnership
            child_prob = [
                10 if child_id in partnership_ids else 1
                for child_id in possible_children_ids
            ]
            child_prob = [x / sum(child_prob) for x in child_prob]
            children_count = min(random.randint(1, 8), len(possible_children_ids))
            # children_ids = random.sample(population=possible_children_ids, k=children_count)

            children_ids = np.random.choice(
                possible_children_ids, size=children_count, p=child_prob, replace=False
            )

            # Add selected children to assigned set
            assigned_children.update(children_ids)

            # Use the first male/nonbinary gendered last name in the relationship as all of the children's last name
            last_name = (
                df_people.loc[df_people["id"].isin(parent_ids)]
                .sort_values(["gender"], ascending=False)["last_name"]
                .iloc[0]
            )
            # print(f"child - {last_name}")
            for parent in parent_ids:
                relationships_people += [
                    {
                        "relationship_id": parent_child_id,
                        "people_id": parent,
                        "title": "parent",
                    }
                ]
            for child_id in children_ids:
                relationships_people += [
                    {
                        "relationship_id": parent_child_id,
                        "people_id": child_id,
                        "title": "child",
                    }
                ]
                # print(df_people.loc[df_people["id"] == child_id, ["first_name","last_name","maiden_name"]].values[0].tolist())
                df_people.loc[df_people["id"] == child_id, "last_name"] = last_name
                df_people.loc[df_people["id"] == child_id, "maiden_name"] = last_name
    return relationships_people, df_people


def generate_tree(
    df_first_names: pd.DataFrame,
    df_surnames: pd.DataFrame,
    df_non_englsh_names: pd.DataFrame = None,
    people: pd.DataFrame = None,
    relationships: pd.DataFrame = None,
    people_count: int = 100,
    generations: int = 0,
):
    if people is None:
        df_people = pd.DataFrame(
            [
                generate_person(
                    df_first_names=df_first_names,
                    df_non_englsh_names=df_non_englsh_names,
                    df_surnames=df_surnames,
                )
                for _ in range(people_count)
            ]
        )
        df_people = df_people.reset_index(names="id")
    if relationships is None:
        relationship_count = int(people_count / 3)
        df_relationships = pd.DataFrame(
            [generate_relationship() for _ in range(relationship_count)]
        )
        df_relationships = df_relationships.reset_index(names="id")

    # Create partnerships
    relationships_people = []
    relationships_people = find_partnerships(
        df_people=df_people,
        df_relationships=df_relationships,
        relationships_people=relationships_people,
    )

    df_relationships_people = pd.DataFrame(relationships_people)

    # Drop partnerships with zero matches
    parternship_ids = df_relationships.loc[
        df_relationships["type"] == "partnership", "id"
    ].tolist()
    match_ids = df_relationships_people["relationship_id"].unique().tolist()
    zero_matches = list(set(parternship_ids).difference(set(match_ids)))
    df_relationships = df_relationships.loc[~df_relationships["id"].isin(zero_matches)]

    # Create children
    if len(relationships_people) > 0:
        relationships_people, df_people = find_children(
            df_people=df_people,
            df_relationships=df_relationships,
            relationships_people=relationships_people,
        )
    df_relationships_people = pd.DataFrame(relationships_people)

    parent_children_ids = df_relationships_people.loc[
        df_relationships_people["title"].isin(["parent", "child"]), "people_id"
    ].tolist()

    # Drop everyone not part of the family tree
    family_ids = df_relationships_people["people_id"].tolist()
    relationship_ids = df_relationships_people["relationship_id"].tolist()
    df_people = df_people.loc[df_people["id"].isin(family_ids)]
    df_relationships = df_relationships.loc[
        df_relationships["id"].isin(relationship_ids)
    ]

    return df_people, df_relationships, df_relationships_people


def prune_tree(
    df_people: pd.DataFrame,
    df_relationships: pd.DataFrame,
    df_relationships_people: pd.DataFrame,
):
    # Keep only the largest connected component of the family tree
    G = nx.Graph()
    for _, rels in df_relationships.iterrows():
        if rels["type"] == "partnership":
            people = df_relationships_people.loc[
                df_relationships_people["relationship_id"] == rels["id"], "people_id"
            ].tolist()
            G.add_edges_from([tuple(people)])
        elif rels["type"] == "parent_child":
            parents = df_relationships_people.loc[
                (df_relationships_people["relationship_id"] == rels["id"])
                & (df_relationships_people["title"] == "parent"),
                "people_id",
            ].tolist()
            children = df_relationships_people.loc[
                (df_relationships_people["relationship_id"] == rels["id"])
                & (df_relationships_people["title"] == "child"),
                "people_id",
            ].tolist()
            for p in parents:
                G.add_edges_from([(p, x) for x in children])

    # Find the largest connected component
    pruned_people = list(max(nx.connected_components(G), key=len))
    df_people = df_people.loc[df_people["id"].isin(pruned_people)]
    df_relationships_people = df_relationships_people.loc[
        df_relationships_people["people_id"].isin(pruned_people)
    ]
    df_relationships = df_relationships.loc[
        df_relationships["id"].isin(df_relationships_people["relationship_id"])
    ]
    return df_people, df_relationships, df_relationships_people


def convert_format_deprecated(df_people, df_relationships_people):
    # Convert the data into the json format used by the family chart visualization
    # this version required one parent to be the mother and one to be the father

    parent_ids = df_relationships_people.loc[
        df_relationships_people["title"] == "parent", "people_id"
    ].tolist()
    children_ids = df_relationships_people.loc[
        df_relationships_people["title"] == "child", "people_id"
    ].tolist()
    partner_ids = df_relationships_people.loc[
        df_relationships_people["title"] == "partner", "people_id"
    ].tolist()
    data_output = []
    for _, people in df_people.iterrows():
        # print(people["id"])
        if people["gender"] not in ["male", "female"]:
            gender = random.sample(["M", "F"], k=1)[0]
        else:
            gender = people["gender"][0].upper()
        df_tmp = {
            "id": str(people["id"]),
            "rels": {},
            "data": {
                "gender": gender,
                "first name": people["first_name"],
                "last name": people["last_name"],
                "birthday": people["date_of_birth"].isoformat(),
                "avatar": "",
            },
        }

        if people["id"] in parent_ids:
            relationship_id = df_relationships_people.loc[
                (df_relationships_people["people_id"] == people["id"])
                & (df_relationships_people["title"] == "parent"),
                "relationship_id",
            ].tolist()[0]
            children = df_relationships_people.loc[
                (df_relationships_people["relationship_id"] == relationship_id)
                & (df_relationships_people["title"] == "child"),
                "people_id",
            ].tolist()
            df_tmp["rels"]["children"] = [str(x) for x in children]

        if people["id"] in children_ids:
            relationship_id = df_relationships_people.loc[
                (df_relationships_people["people_id"] == people["id"])
                & (df_relationships_people["title"] == "child"),
                "relationship_id",
            ].tolist()[0]
            parents = df_relationships_people.loc[
                (df_relationships_people["relationship_id"] == relationship_id)
                & (df_relationships_people["title"] == "parent"),
                "people_id",
            ].tolist()
            # Current versin of visualization requires one parent to be the father and one to be the mother,
            # so using the one who changed their last name to be the mother
            # queer relationships break this logic so for that we'll just pick them at random

            try:
                if (
                    len(
                        set(
                            df_people.loc[df_people["id"].isin(parents), "sex"].tolist()
                        )
                    )
                    == 1
                ):
                    parental_roles = random.sample(parents, k=2)
                    mother_id = parental_roles[0]
                    father_id = parental_roles[1]
                else:
                    mother_id = df_people.loc[
                        (df_people["id"].isin(parents))
                        & (df_people["sex"] == "female"),
                        "id",
                    ].tolist()[0]
                    father_id = df_people.loc[
                        (df_people["id"].isin(parents)) & (df_people["sex"] == "male"),
                        "id",
                    ].tolist()[0]
            except IndexError as e:
                print(f"parents: {parents}")
                raise e
            df_tmp["rels"]["mother"] = str(mother_id)
            df_tmp["rels"]["father"] = str(father_id)

        if people["id"] in partner_ids:
            relationship_id = df_relationships_people.loc[
                (df_relationships_people["people_id"] == people["id"])
                & (df_relationships_people["title"] == "partner"),
                "relationship_id",
            ].tolist()[0]
            partners = df_relationships_people.loc[
                (df_relationships_people["relationship_id"] == relationship_id)
                & (df_relationships_people["title"] == "partner")
                & (df_relationships_people["people_id"] != people["id"]),
                "people_id",
            ].tolist()
            df_tmp["rels"]["spouses"] = [str(x) for x in partners]

        data_output.append(df_tmp)
    return data_output


def convert_format(df_people, df_relationships_people):
    # Convert the data into the json format used by the family chart visualization
    # this version required one parent to be the mother and one to be the father

    parent_ids = df_relationships_people.loc[
        df_relationships_people["title"] == "parent", "people_id"
    ].tolist()
    children_ids = df_relationships_people.loc[
        df_relationships_people["title"] == "child", "people_id"
    ].tolist()
    partner_ids = df_relationships_people.loc[
        df_relationships_people["title"] == "partner", "people_id"
    ].tolist()
    data_output = []
    for _, people in df_people.iterrows():
        # print(people["id"])
        if people["gender"] not in ["male", "female"]:
            gender = random.sample(["M", "F"], k=1)[0]
        else:
            gender = people["gender"][0].upper()
        df_tmp = {
            "id": str(people["id"]),
            "rels": {},
            "data": {
                "gender": gender,
                "first name": people["first_name"],
                "last name": people["last_name"],
                "birthday": people["date_of_birth"].isoformat(),
                "avatar": "",
            },
        }

        if people["id"] in parent_ids:
            relationship_id = df_relationships_people.loc[
                (df_relationships_people["people_id"] == people["id"])
                & (df_relationships_people["title"] == "parent"),
                "relationship_id",
            ].tolist()[0]
            children = df_relationships_people.loc[
                (df_relationships_people["relationship_id"] == relationship_id)
                & (df_relationships_people["title"] == "child"),
                "people_id",
            ].tolist()
            df_tmp["rels"]["children"] = [str(x) for x in children]

        if people["id"] in children_ids:
            relationship_id = df_relationships_people.loc[
                (df_relationships_people["people_id"] == people["id"])
                & (df_relationships_people["title"] == "child"),
                "relationship_id",
            ].tolist()[0]
            parents = df_relationships_people.loc[
                (df_relationships_people["relationship_id"] == relationship_id)
                & (df_relationships_people["title"] == "parent"),
                "people_id",
            ].tolist()
            df_tmp["rels"]["parents"] = [str(x) for x in parents]


        if people["id"] in partner_ids:
            relationship_id = df_relationships_people.loc[
                (df_relationships_people["people_id"] == people["id"])
                & (df_relationships_people["title"] == "partner"),
                "relationship_id",
            ].tolist()[0]
            partners = df_relationships_people.loc[
                (df_relationships_people["relationship_id"] == relationship_id)
                & (df_relationships_people["title"] == "partner")
                & (df_relationships_people["people_id"] != people["id"]),
                "people_id",
            ].tolist()
            df_tmp["rels"]["spouses"] = [str(x) for x in partners]

        data_output.append(df_tmp)
    return data_output
