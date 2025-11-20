# monday_extract_groups.py
from __future__ import annotations
import os
import requests
import sys
import csv
from tqdm import tqdm
import pandas as pd
import re
import warnings
import logging
import pandas as pd
import numpy as np

from dataclasses import dataclass
from datetime import date
from typing import Callable, Final, Tuple

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype




logger = logging.getLogger(__name__)


def fetch_groups(board_id, api_key):
    """
    Fetches groups from a specified Monday.com board.

    Args:
        board_id (str): The ID of the board.
        api_key (str): Your Monday.com API key.

    Returns:
        list: A list of groups with their IDs and titles.
    """
    query = """
    query ($boardId: [ID!]!) {
      boards(ids: $boardId) {
        groups {
          id
          title
        }
      }
    }
    """

    variables = {"boardId": [str(board_id)]}

    headers = {
        'Authorization': f'Bearer {api_key}',
        'Content-Type': 'application/json'
    }

    response = requests.post("https://api.monday.com/v2",
                             json={
                                 "query": query,
                                 "variables": variables
                             },
                             headers=headers)

    if response.status_code != 200:
        raise Exception(
            f"Query failed with status code {response.status_code}: {response.text}"
        )

    data = response.json()

    if 'errors' in data:
        error_messages = "\n".join(
            [error['message'] for error in data['errors']])
        raise Exception(f"GraphQL Errors:\n{error_messages}")

    boards = data.get('data', {}).get('boards', [])
    if not boards:
        raise Exception(f"No boards found with ID {board_id}.")

    board = boards[0]
    groups = board.get('groups', [])

    if not groups:
        raise Exception(f"No groups found in board {board_id}.")

    return groups


def fetch_items(board_id, group_id, api_key, limit=10):
    """
    Fetches items from a specific group within a Monday.com board.

    Args:
        board_id (str): The ID of the board.
        group_id (str): The ID of the group.
        api_key (str): Your Monday.com API key.
        limit (int): Number of items to fetch.

    Returns:
        list: A list of items with their details.
    """
    query = """
    query ($boardId: [ID!]!, $groupId: [String!]!, $limit: Int!) {
      boards(ids: $boardId) {
        groups(ids: $groupId) {
          id
          title
          items_page(limit: $limit) {
            items {
              id
              name
              column_values {
                id
                text
              }
            }
          }
        }
      }
    }
    """

    variables = {
        "boardId": [str(board_id)],
        "groupId": [str(group_id)],
        "limit": limit
    }

    headers = {
        'Authorization': f'Bearer {api_key}',
        'Content-Type': 'application/json'
    }

    response = requests.post("https://api.monday.com/v2",
                             json={
                                 "query": query,
                                 "variables": variables
                             },
                             headers=headers)

    if response.status_code != 200:
        raise Exception(
            f"Query failed with status code {response.status_code}: {response.text}"
        )

    data = response.json()

    if 'errors' in data:
        error_messages = "\n".join(
            [error['message'] for error in data['errors']])
        raise Exception(f"GraphQL Errors:\n{error_messages}")

    boards = data.get('data', {}).get('boards', [])
    if not boards:
        raise Exception(f"No boards found with ID {board_id}.")

    board = boards[0]
    groups = board.get('groups', [])
    if not groups:
        raise Exception(
            f"No groups found with ID '{group_id}' in board {board_id}.")

    group = groups[0]
    items_page = group.get('items_page', {})
    items = items_page.get('items', [])

    return items


def export_items_to_csv(items, filename):
    """
    Exports fetched items to a CSV file.

    Args:
        items (list): List of items to export.
        filename (str): The name of the CSV file.
    """
    if not items:
        return

    headers = ['Item ID', 'Item Name']
    column_ids = []
    for column in items[0]['column_values']:
        headers.append(column['id'])
        column_ids.append(column['id'])

    with open(filename, mode='w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=headers)
        writer.writeheader()

        for item in items:
            row = {'Item ID': item['id'], 'Item Name': item['name']}
            for column in item['column_values']:
                row[column['id']] = column.get('text', '')
            writer.writerow(row)


def fetch_items_recursive(board_id, group_id, api_key, limit=500):
    """
    Recursively fetches all items from a specific group within a Monday.com board using cursor-based pagination.

    Args:
        board_id (str): The ID of the board.
        group_id (str): The ID of the group.
        api_key (str): Your Monday.com API key.
        limit (int, optional): Number of items to fetch per request. Defaults to 500.

    Returns:
        list: A complete list of all items in the group.
    """
    all_items = []
    headers = {
        'Authorization': f'Bearer {api_key}',
        'Content-Type': 'application/json'
    }

    initial_query = """
    query ($boardId: [ID!]!, $groupId: [String!]!, $limit: Int!) {
      boards(ids: $boardId) {
        groups(ids: $groupId) {
          id
          title
          items_page(limit: $limit) {
            cursor
            items {
              id
              name
              column_values {
                id
                text
              }
            }
          }
        }
      }
    }
    """

    variables = {
        "boardId": [str(board_id)],
        "groupId": [str(group_id)],
        "limit": limit
    }

    response = requests.post("https://api.monday.com/v2",
                             json={
                                 "query": initial_query,
                                 "variables": variables
                             },
                             headers=headers)

    if response.status_code != 200:
        raise Exception(
            f"Initial query failed with status code {response.status_code}: {response.text}"
        )

    data = response.json()

    if 'errors' in data:
        error_messages = "\n".join(
            [error['message'] for error in data['errors']])
        raise Exception(f"GraphQL Errors in initial query:\n{error_messages}")

    try:
        group = data['data']['boards'][0]['groups'][0]
        items_page = group.get('items_page', {})
        items = items_page.get('items', [])
        all_items.extend(items)
        cursor = items_page.get('cursor')
    except (IndexError, KeyError) as e:
        raise Exception(f"Error parsing initial response: {e}")

    while cursor:
        next_query = """
        query ($limit: Int!, $cursor: String!) {
          next_items_page(limit: $limit, cursor: $cursor) {
            cursor
            items {
              id
              name
              column_values {
                id
                text
              }
            }
          }
        }
        """

        next_variables = {"limit": limit, "cursor": cursor}

        response = requests.post("https://api.monday.com/v2",
                                 json={
                                     "query": next_query,
                                     "variables": next_variables
                                 },
                                 headers=headers)

        if response.status_code != 200:
            raise Exception(
                f"Next items query failed with status code {response.status_code}: {response.text}"
            )

        data = response.json()

        if 'errors' in data:
            error_messages = "\n".join(
                [error['message'] for error in data['errors']])
            raise Exception(
                f"GraphQL Errors in next_items_page query:\n{error_messages}")

        try:
            next_page = data['data']['next_items_page']
            items = next_page.get('items', [])
            all_items.extend(items)
            cursor = next_page.get('cursor')
        except (KeyError, TypeError) as e:
            raise Exception(f"Error parsing next_items_page response: {e}")

    return all_items


def fetch_and_export_all_groups(board_id,
                                group_list,
                                name_list,
                                api_key,
                                limit=500):
    """
    Fetches items from all specified groups and exports them to corresponding CSV files.

    Args:
        board_id (str): The ID of the board.
        group_list (list): List of group IDs to fetch.
        name_list (list): List of filenames for each group.
        api_key (str): Your Monday.com API key.
        limit (int, optional): Number of items to fetch per request. Defaults to 500.
    """
    groups = fetch_groups(board_id, api_key)
    group_dict = {group['id']: group for group in groups}

    for group_id, filename in tqdm(zip(group_list, name_list),
                                   total=len(group_list),
                                   desc="Fetching Groups"):
        if group_id not in group_dict:
            # Optionally, handle missing groups as needed
            continue

        items = fetch_items_recursive(board_id, group_id, api_key, limit)
        export_items_to_csv(items, filename)


date_pattern = r"\d{4}-\d{2}-\d{2}"


def extract_date(value):
    """
    Extracts date from a string using a regex pattern.
    """
    if pd.isna(value) or value == 'NaT':
        return None  # Handle NaT or NaN values
    if isinstance(value, str):
        match = re.search(date_pattern, value)
        return match.group(0) if match else None
    return None


def items_to_dataframe(items):
    """
    Converts a list of items to a pandas DataFrame.
    """
    if not items:
        logger.warning("No items to convert.")
        return pd.DataFrame()

    data = []
    column_ids = [column['id'] for column in items[0]['column_values']]
    headers = ['Item ID', 'Item Name'] + column_ids

    for item in items:
        row = {'Item ID': item['id'], 'Item Name': item['name']}
        for column in item['column_values']:
            row[column['id']] = column.get('text', '')
        data.append(row)

    df = pd.DataFrame(data, columns=headers)
    return df


def fetch_data():
    """
    Fetches data from Monday.com and returns a dictionary of DataFrames.
    """
    BOARD_ID = "6942829967"
    group_list = [
        "topics", "new_group34578__1", "new_group27351__1",
        "new_group54376__1", "new_group64021__1", "new_group65903__1",
        "new_group62617__1"
    ]
    name_list = [
        "scheduled", "unqualified", "won", "cancelled", "noshow", "proposal",
        "lost"
    ]
    LIMIT = 500  # Items limit per group

    # Fetch API key from secrets
    try:
        api_key = os.getenv("MONDAY_API_KEY")
    except KeyError:
        logger.error("Error: MONDAY_API_KEY is not set in .env.")

    # Fetch all groups from the board
    try:
        api_key = os.getenv("MONDAY_API_KEY")
        groups = fetch_groups(BOARD_ID, api_key)
    except Exception as e:
        logger.error(f"Error fetching groups: {e}")

    dataframes = {}

    total_groups = len(group_list)
    progress_percentage = 0.0
    progress_step = 100 / total_groups

    for i, (group_id, group_name) in tqdm(enumerate(zip(group_list,
                                                        name_list))):
        # Find the target group
        target_group = next(
            (group for group in groups if group['id'] == group_id), None)
        if not target_group:
            logger.error(
                f"Group with ID '{group_id}' not found in board {BOARD_ID}.")

        print(
            f"Fetching items from Group: **{target_group['title']}** (ID: {target_group['id']})"
        )

        # Fetch items from the target group
        try:
            items = fetch_items_recursive(BOARD_ID, target_group['id'],
                                          api_key, LIMIT)
        except Exception as e:
            logger.error(f"Error fetching items for group '{group_name}': {e}")

        df_items = items_to_dataframe(items)
        dataframes[group_name] = df_items

        # Update progress percentage
        progress_percentage += progress_step
        print(f"Progress: {progress_percentage:.2f}%")

    # Define column renaming mapping
    columns_with_titles = {
        'name': 'Name',
        'auto_number__1': 'Auto number',
        'person': 'Owner',
        'last_updated__1': 'Last updated',
        'link__1': 'Linkedin',
        'phone__1': 'Phone',
        'email__1': 'Email',
        'text7__1': 'Company',
        'date4': 'Sales Call Date',
        'status9__1': 'Follow Up Tracker',
        'notes__1': 'Notes',
        'interested_in__1': 'Interested In',
        'status4__1': 'Plan Type',
        'numbers__1': 'Deal Value',
        'status6__1': 'Email Template #1',
        'dup__of_email_template__1': 'Email Template #2',
        'status__1': 'Deal Status',
        'status2__1': 'Send Panda Doc?',
        'utm_source__1': 'UTM Source',
        'date__1': 'Deal Status Date',
        'utm_campaign__1': 'UTM Campaign',
        'utm_medium__1': 'UTM Medium',
        'utm_content__1': 'UTM Content',
        'link3__1': 'UTM LINK',
        'lead_source8__1': 'Lead Source',
        'color__1': 'Channel FOR FUNNEL METRICS',
        'subitems__1': 'Subitems',
        'date5__1': 'Date Created'
    }

    # Rename columns in each dataframe
    for key in dataframes.keys():
        df = dataframes[key]
        df.rename(columns=columns_with_titles, inplace=True)
        dataframes[key] = df

    return dataframes


# ──────────────────────────────────────────────────────────────────────────────
#  Re-implemented process_data – production-ready, no NaN/None in rate columns
# ──────────────────────────────────────────────────────────────────────────────
# def process_data(dataframes: dict[str, pd.DataFrame], st_date: str,
#                  end_date: str, filter_column: str) -> pd.DataFrame:
#     """
#     Build the KPI table for the date range [st_date, end_date] (inclusive).

#     Parameters
#     ----------
#     dataframes     output of fetch_data(); keys such as 'scheduled', 'won', …
#     st_date        'YYYY-MM-DD' – range start
#     end_date       'YYYY-MM-DD' – range end
#     filter_column  column chosen in the UI for date filtering
#                    (usually 'Date Created' , 'Sales Call Date' , '')
#     """
#     # ── unpack individual stages (empty DF if missing) ────────────────────
#     op_cancelled = dataframes.get("cancelled", pd.DataFrame())
#     op_lost = dataframes.get("lost", pd.DataFrame())
#     op_noshow = dataframes.get("noshow", pd.DataFrame())
#     op_proposal = dataframes.get("proposal", pd.DataFrame())
#     op_scheduled = dataframes.get("scheduled", pd.DataFrame())
#     op_unqualified = dataframes.get("unqualified", pd.DataFrame())
#     op_won = dataframes.get("won", pd.DataFrame())

#     # ── canonical list of owners  (whitespace already normalised in fetch_data)
#     owners = (pd.concat([
#         op_cancelled, op_lost, op_noshow, op_proposal, op_scheduled,
#         op_unqualified, op_won
#     ])["Owner"].dropna().unique())
#     kpi = pd.DataFrame(index=owners)
#     kpi.index.name = "Owner"
#     kpi["Owner"] = kpi.index  # explicit column for display

#     # ── convenience: date-range filter  ───────────────────────────────────
#     def _filter(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
#         if date_col not in df.columns:
#             return pd.DataFrame(columns=df.columns)

#         dates = pd.to_datetime(df[date_col].apply(extract_date),
#                                errors="coerce").dt.date
#         mask = ((dates >= pd.to_datetime(st_date).date()) &
#                 (dates <= pd.to_datetime(end_date).date()))
#         return df.loc[mask]

#     fdate = _filter  # alias

#     # ───────────────────────────────────────────────────────────────────────
#     #  RAW COUNTS
#     # ───────────────────────────────────────────────────────────────────────
#     all_stages = pd.concat([
#         op_cancelled, op_lost, op_noshow, op_proposal, op_scheduled,
#         op_unqualified, op_won
#     ])
#     kpi["New Calls Booked"] = (fdate(
#         all_stages,
#         filter_column).groupby("Owner").size().reindex(kpi.index,
#                                                        fill_value=0))

#     sc_taken_df = pd.concat([op_unqualified, op_proposal, op_won, op_lost])
#     kpi["Sales Call Taken"] = (fdate(
#         sc_taken_df,
#         filter_column).groupby("Owner").size().reindex(kpi.index,
#                                                        fill_value=0))

#     kpi["Unqualified"] = (fdate(op_unqualified,
#                                 filter_column).groupby("Owner").size().reindex(
#                                     kpi.index, fill_value=0))

#     kpi["Cancelled Calls"] = (fdate(
#         op_cancelled,
#         filter_column).groupby("Owner").size().reindex(kpi.index,
#                                                        fill_value=0))

#     # Proposals count = Proposal + Won + Lost (anchor to Sales Call date if present)
#     prop_date_col = ("Sales Call Date" if "Sales Call Date"
#                      in op_proposal.columns else filter_column)
#     kpi["Proposals"] = (pd.concat([op_proposal, op_won, op_lost]).pipe(
#         lambda df: fdate(df, prop_date_col)).groupby("Owner").size().reindex(
#             kpi.index, fill_value=0))

#     # ───────────────────────────────────────────────────────────────────────
#     #  RATE METRICS  (all numeric, no NaN/None)
#     # ───────────────────────────────────────────────────────────────────────
#     with np.errstate(divide="ignore", invalid="ignore"):
#         kpi["Show Rate %"] = (kpi["Sales Call Taken"] / kpi["New Calls Booked"]
#                               ).replace([np.inf, -np.inf], 0).fillna(0) * 100

#         kpi["Unqualified Rate %"] = (kpi["Unqualified"] /
#                                      kpi["New Calls Booked"]).replace(
#                                          [np.inf, -np.inf], 0).fillna(0) * 100

#         kpi["Cancellation Rate %"] = (kpi["Cancelled Calls"] /
#                                       kpi["New Calls Booked"]).replace(
#                                           [np.inf, -np.inf], 0).fillna(0) * 100

#         kpi["Proposal Rate %"] = (kpi["Proposals"] /
#                                   kpi["New Calls Booked"]).replace(
#                                       [np.inf, -np.inf], 0).fillna(0) * 100

#     # Close metrics
#     closes = fdate(op_won, prop_date_col).groupby("Owner").size()
#     kpi["Close"] = closes.reindex(kpi.index, fill_value=0)  # helper column

#     with np.errstate(divide="ignore", invalid="ignore"):
#         kpi["Close Rate %"] = (kpi["Close"] / kpi["New Calls Booked"]).replace(
#             [np.inf, -np.inf], 0).fillna(0) * 100

#         kpi["Close Rate(Show) %"] = (kpi["Close"] /
#                                      kpi["Sales Call Taken"]).replace(
#                                          [np.inf, -np.inf], 0).fillna(0) * 100

#         kpi["Close Rate(MQL) %"] = (kpi["Close"] / kpi["Proposals"].replace(
#             0, np.nan)).replace([np.inf, -np.inf], 0).fillna(0) * 100

#     # ───────────────────────────────────────────────────────────────────────
#     #  REVENUE METRICS
#     # ───────────────────────────────────────────────────────────────────────
#     # Always use the user-selected filter_column for consistency
#     won_rev = fdate(op_won.copy(), filter_column)
#     won_rev["Deal Value"] = pd.to_numeric(won_rev["Deal Value"],
#                                           errors="coerce").fillna(0)

#     rev_sum = won_rev.groupby("Owner")["Deal Value"].sum()
#     kpi["Closed Revenue $"] = rev_sum.reindex(kpi.index, fill_value=0)

#     with np.errstate(divide="ignore", invalid="ignore"):
#         kpi["Revenue Per Call $"] = (kpi["Closed Revenue $"] /
#                                      kpi["New Calls Booked"]).replace(
#                                          [np.inf, -np.inf], 0).fillna(0)

#         kpi["Revenue Per Showed Up $"] = (kpi["Closed Revenue $"] /
#                                           kpi["Sales Call Taken"]).replace(
#                                               [np.inf, -np.inf], 0).fillna(0)

#         kpi["Revenue Per Proposal $"] = (
#             kpi["Closed Revenue $"] /
#             kpi["Proposals"].replace(0, np.nan)).replace([np.inf, -np.inf],
#                                                          0).fillna(0)

#     # Pipeline revenue (open proposals)
#     pipe_rev = fdate(op_proposal.copy(), prop_date_col)
#     pipe_rev["Deal Value"] = pd.to_numeric(pipe_rev["Deal Value"],
#                                            errors="coerce").fillna(0)
#     kpi["Pipeline Revenue $"] = (
#         pipe_rev.groupby("Owner")["Deal Value"].sum().reindex(kpi.index,
#                                                               fill_value=0))

#     # ── TOTAL ROW  ──────────────────────────────────────────────────────────
#     totals = {
#         "Owner": "Total",
#         "New Calls Booked": kpi["New Calls Booked"].sum(),
#         "Sales Call Taken": kpi["Sales Call Taken"].sum(),
#         "Proposals": kpi["Proposals"].sum(),
#         "Show Rate %": kpi["Show Rate %"].mean(),
#         "Unqualified": kpi["Unqualified"].sum(),
#         "Unqualified Rate %": kpi["Unqualified Rate %"].mean(),
#         "Cancelled Calls": kpi["Cancelled Calls"].sum(),
#         "Cancellation Rate %": kpi["Cancellation Rate %"].mean(),
#         "Proposal Rate %": kpi["Proposal Rate %"].mean(),
#         "Close Rate %": kpi["Close Rate %"].mean(),
#         "Close Rate(Show) %": kpi["Close Rate(Show) %"].mean(),
#         "Close Rate(MQL) %": kpi["Close Rate(MQL) %"].mean(),
#         "Closed Revenue $": kpi["Closed Revenue $"].sum(),
#         "Revenue Per Call $": kpi["Revenue Per Call $"].mean(),
#         "Revenue Per Showed Up $": kpi["Revenue Per Showed Up $"].mean(),
#         "Revenue Per Proposal $": kpi["Revenue Per Proposal $"].mean(),
#         "Pipeline Revenue $": kpi["Pipeline Revenue $"].sum(),
#     }

#     kpi_final = (pd.concat([kpi,
#                             pd.DataFrame([totals]).set_index("Owner")
#                             ]).reset_index(drop=True)).drop(columns=["Close"],
#                                                             errors="ignore")

#     return kpi_final


# def process_data(dataframes: dict[str, pd.DataFrame], st_date: str,
#                  end_date: str, filter_column: str) -> pd.DataFrame:
#     """
#     Build the KPI table for the date range [st_date, end_date] (inclusive).

#     Parameters
#     ----------
#     dataframes     output of fetch_data(); keys such as 'scheduled', 'won', …
#     st_date        'YYYY-MM-DD' – range start
#     end_date       'YYYY-MM-DD' – range end
#     filter_column  column chosen in the UI for date filtering
#            (usually 'Date Created' , 'Sales Call Date' , '')
#     """
#     # ── unpack individual stages (empty DF if missing) ────────────────────
#     op_cancelled = dataframes.get("cancelled", pd.DataFrame())
#     op_lost = dataframes.get("lost", pd.DataFrame())
#     op_noshow = dataframes.get("noshow", pd.DataFrame())
#     op_proposal = dataframes.get("proposal", pd.DataFrame())
#     op_scheduled = dataframes.get("scheduled", pd.DataFrame())
#     op_unqualified = dataframes.get("unqualified", pd.DataFrame())
#     op_won = dataframes.get("won", pd.DataFrame())

#     # ── canonical list of owners  (whitespace already normalised in fetch_data)
#     owners = (pd.concat([
#         op_cancelled, op_lost, op_noshow, op_proposal, op_scheduled,
#         op_unqualified, op_won
#     ])["Owner"].dropna().unique())
#     kpi = pd.DataFrame(index=owners)
#     kpi.index.name = "Owner"
#     kpi["Owner"] = kpi.index  # explicit column for display

#     # ── convenience: date-range filter  ───────────────────────────────────
#     def _filter(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
#         if date_col not in df.columns:
#             return pd.DataFrame(columns=df.columns)

#         dates = pd.to_datetime(df[date_col].apply(extract_date),
#                                errors="coerce").dt.date
#         mask = ((dates >= pd.to_datetime(st_date).date()) &
#                 (dates <= pd.to_datetime(end_date).date()))
#         return df.loc[mask]

#     fdate = _filter  # alias

#     # ───────────────────────────────────────────────────────────────────────
#     #  RAW COUNTS
#     # ───────────────────────────────────────────────────────────────────────
#     all_stages = pd.concat([
#         op_cancelled, op_lost, op_noshow, op_proposal, op_scheduled,
#         op_unqualified, op_won
#     ])
#     kpi["New Calls Booked"] = (fdate(
#         all_stages,
#         filter_column).groupby("Owner").size().reindex(kpi.index,
#                                                        fill_value=0))

#     sc_taken_df = pd.concat([op_unqualified, op_proposal, op_won, op_lost])
#     kpi["Sales Call Taken"] = (fdate(
#         sc_taken_df,
#         filter_column).groupby("Owner").size().reindex(kpi.index,
#                                                        fill_value=0))

#     kpi["Unqualified"] = (fdate(op_unqualified,
#                                 filter_column).groupby("Owner").size().reindex(
#                                     kpi.index, fill_value=0))

#     kpi["Cancelled Calls"] = (fdate(
#         op_cancelled,
#         filter_column).groupby("Owner").size().reindex(kpi.index,
#                                                        fill_value=0))

#     # Proposals count = Proposal + Won + Lost (anchor to Sales Call date if present)
#     prop_date_col = ("Sales Call Date" if "Sales Call Date"
#                      in op_proposal.columns else filter_column)
#     kpi["Proposals"] = (pd.concat([op_proposal, op_won, op_lost]).pipe(
#         lambda df: fdate(df, prop_date_col)).groupby("Owner").size().reindex(
#             kpi.index, fill_value=0))

#     # ───────────────────────────────────────────────────────────────────────
#     #  Origin Counts
#     # ───────────────────────────────────────────────────────────────────────
#     #Count the number of cold emails in origin column and group by owner
#     #cold_emails = all_stages[all_stages['origin'] == 'cold-email']
#     kpi["Cold Emails"] = (fdate(
#         all_stages[all_stages['origin'] == 'cold-email'],
#         filter_column).groupby("Owner").size().reindex(kpi.index,
#                                                        fill_value=0))

#     #Count the number of linkedin in origin column and group by owner
#     # linkedin = all_stages[all_stages['origin'] == 'LinkedIn']
#     # kpi["LinkedIn"] = linkedin.groupby("Owner").size().reindex(kpi.index,
#     #                                                            fill_value=0)
#     kpi["LinkedIn"] = np.where(kpi['New Calls Booked'] > 0,
#                                kpi['New Calls Booked'] - kpi['Cold Emails'], 0)

#     # ───────────────────────────────────────────────────────────────────────
#     #  RATE METRICS  (all numeric, no NaN/None)
#     # ───────────────────────────────────────────────────────────────────────
#     with np.errstate(divide="ignore", invalid="ignore"):
#         kpi["Show Rate %"] = (kpi["Sales Call Taken"] / kpi["New Calls Booked"]
#                               ).replace([np.inf, -np.inf], 0).fillna(0) * 100

#         kpi["Unqualified Rate %"] = (kpi["Unqualified"] /
#                                      kpi["New Calls Booked"]).replace(
#                                          [np.inf, -np.inf], 0).fillna(0) * 100

#         kpi["Cancellation Rate %"] = (kpi["Cancelled Calls"] /
#                                       kpi["New Calls Booked"]).replace(
#                                           [np.inf, -np.inf], 0).fillna(0) * 100

#         kpi["Proposal Rate %"] = (kpi["Proposals"] /
#                                   kpi["New Calls Booked"]).replace(
#                                       [np.inf, -np.inf], 0).fillna(0) * 100

#     # Close metrics
#     closes = fdate(op_won, prop_date_col).groupby("Owner").size()
#     kpi["Close"] = closes.reindex(kpi.index, fill_value=0)  # helper column

#     with np.errstate(divide="ignore", invalid="ignore"):
#         kpi["Close Rate %"] = (kpi["Close"] / kpi["New Calls Booked"]).replace(
#             [np.inf, -np.inf], 0).fillna(0) * 100

#         kpi["Close Rate(Show) %"] = (kpi["Close"] /
#                                      kpi["Sales Call Taken"]).replace(
#                                          [np.inf, -np.inf], 0).fillna(0) * 100

#         kpi["Close Rate(MQL) %"] = (kpi["Close"] / kpi["Proposals"].replace(
#             0, np.nan)).replace([np.inf, -np.inf], 0).fillna(0) * 100

#     # ───────────────────────────────────────────────────────────────────────
#     #  REVENUE METRICS
#     # ───────────────────────────────────────────────────────────────────────
#     # Always use the user-selected filter_column for consistency
#     won_rev = fdate(op_won.copy(), filter_column)
#     won_rev["Deal Value"] = pd.to_numeric(won_rev["Deal Value"],
#                                           errors="coerce").fillna(0)

#     rev_sum = won_rev.groupby("Owner")["Deal Value"].sum()
#     kpi["Closed Revenue $"] = rev_sum.reindex(kpi.index, fill_value=0)

#     with np.errstate(divide="ignore", invalid="ignore"):
#         kpi["Revenue Per Call $"] = (kpi["Closed Revenue $"] /
#                                      kpi["New Calls Booked"]).replace(
#                                          [np.inf, -np.inf], 0).fillna(0)

#         kpi["Revenue Per Showed Up $"] = (kpi["Closed Revenue $"] /
#                                           kpi["Sales Call Taken"]).replace(
#                                               [np.inf, -np.inf], 0).fillna(0)

#         kpi["Revenue Per Proposal $"] = (
#             kpi["Closed Revenue $"] /
#             kpi["Proposals"].replace(0, np.nan)).replace([np.inf, -np.inf],
#                                                          0).fillna(0)

#     # Pipeline revenue (open proposals)
#     pipe_rev = fdate(op_proposal.copy(), prop_date_col)
#     pipe_rev["Deal Value"] = pd.to_numeric(pipe_rev["Deal Value"],
#                                            errors="coerce").fillna(0)
#     kpi["Pipeline Revenue $"] = (
#         pipe_rev.groupby("Owner")["Deal Value"].sum().reindex(kpi.index,
#                                                               fill_value=0))

#     # ── TOTAL ROW  ────────────────────────────────────────────────────
#     totals = {
#         "Owner": "Total",
#         "New Calls Booked": kpi["New Calls Booked"].sum(),
#         "Sales Call Taken": kpi["Sales Call Taken"].sum(),
#         "Proposals": kpi["Proposals"].sum(),
#         "Show Rate %": kpi["Show Rate %"].mean(),
#         "Unqualified": kpi["Unqualified"].sum(),
#         "Unqualified Rate %": kpi["Unqualified Rate %"].mean(),
#         "Cancelled Calls": kpi["Cancelled Calls"].sum(),
#         "Cancellation Rate %": kpi["Cancellation Rate %"].mean(),
#         "Proposal Rate %": kpi["Proposal Rate %"].mean(),
#         "Close Rate %": kpi["Close Rate %"].mean(),
#         "Close Rate(Show) %": kpi["Close Rate(Show) %"].mean(),
#         "Close Rate(MQL) %": kpi["Close Rate(MQL) %"].mean(),
#         "Closed Revenue $": kpi["Closed Revenue $"].sum(),
#         "Revenue Per Call $": kpi["Revenue Per Call $"].mean(),
#         "Revenue Per Showed Up $": kpi["Revenue Per Showed Up $"].mean(),
#         "Revenue Per Proposal $": kpi["Revenue Per Proposal $"].mean(),
#         "Pipeline Revenue $": kpi["Pipeline Revenue $"].sum(),
#         "Cold Emails": kpi["Cold Emails"].sum(),
#         "LinkedIn": kpi["LinkedIn"].sum(),
#     }

#     # Create totals row with proper index
#     totals_df = pd.DataFrame([totals])

#     # Concatenate the main KPI data with totals
#     kpi_final = pd.concat([kpi.reset_index(drop=True), totals_df],
#                           ignore_index=True).drop(columns=["Close"],
#                                                   errors="ignore")
#     return kpi_final


# def process_data_COLD_EMAIL(dataframes: dict[str, pd.DataFrame], st_date: str,
#                             end_date: str, filter_column: str) -> pd.DataFrame:
#     """
#     Build the KPI table for the date range [st_date, end_date] (inclusive).
    
#     Parameters
#     ----------
#     dataframes     output of fetch_data(); keys such as 'scheduled', 'won', …
#     st_date        'YYYY-MM-DD' – range start
#     end_date       'YYYY-MM-DD' – range end
#     filter_column  column chosen in the UI for date filtering
#     (usually 'Date Created' , 'Sales Call Date' , '')
#     """
#     # ── unpack individual stages (empty DF if missing) ────────────────────
#     op_cancelled = dataframes.get("cancelled", pd.DataFrame())
#     op_lost = dataframes.get("lost", pd.DataFrame())
#     op_noshow = dataframes.get("noshow", pd.DataFrame())
#     op_proposal = dataframes.get("proposal", pd.DataFrame())
#     op_scheduled = dataframes.get("scheduled", pd.DataFrame())
#     op_unqualified = dataframes.get("unqualified", pd.DataFrame())
#     op_won = dataframes.get("won", pd.DataFrame())

#     # ── canonical list of owners  (whitespace already normalised in fetch_data)
#     owners = (pd.concat([
#         op_cancelled, op_lost, op_noshow, op_proposal, op_scheduled,
#         op_unqualified, op_won
#     ])["Owner"].dropna().unique())
#     kpi = pd.DataFrame(index=owners)
#     kpi.index.name = "Owner"
#     kpi["Owner"] = kpi.index  # explicit column for display

#     # ── convenience: date-range filter  ───────────────────────────────────
#     def _filter(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
#         if date_col not in df.columns:
#             return pd.DataFrame(columns=df.columns)

#         dates = pd.to_datetime(df[date_col].apply(extract_date),
#                                errors="coerce").dt.date
#         mask = ((dates >= pd.to_datetime(st_date).date()) &
#                 (dates <= pd.to_datetime(end_date).date()))
#         return df.loc[mask]

#     fdate = _filter  # alias

#     # ───────────────────────────────────────────────────────────────────────
#     #  RAW COUNTS
#     # ───────────────────────────────────────────────────────────────────────
#     all_stages = pd.concat([
#         op_cancelled, op_lost, op_noshow, op_proposal, op_scheduled,
#         op_unqualified, op_won
#     ])
#     kpi["New Calls Booked"] = (fdate(
#         all_stages,
#         filter_column).groupby("Owner").size().reindex(kpi.index,
#                                                        fill_value=0))

#     sc_taken_df = pd.concat([op_unqualified, op_proposal, op_won, op_lost])
#     kpi["Sales Call Taken"] = (fdate(
#         sc_taken_df,
#         filter_column).groupby("Owner").size().reindex(kpi.index,
#                                                        fill_value=0))

#     kpi["Unqualified"] = (fdate(op_unqualified,
#                                 filter_column).groupby("Owner").size().reindex(
#                                     kpi.index, fill_value=0))

#     kpi["Cancelled Calls"] = (fdate(
#         op_cancelled,
#         filter_column).groupby("Owner").size().reindex(kpi.index,
#                                                        fill_value=0))

#     # Proposals count = Proposal + Won + Lost (anchor to Sales Call date if present)
#     prop_date_col = ("Sales Call Date" if "Sales Call Date"
#                      in op_proposal.columns else filter_column)
#     kpi["Proposals"] = (pd.concat([op_proposal, op_won, op_lost]).pipe(
#         lambda df: fdate(df, prop_date_col)).groupby("Owner").size().reindex(
#             kpi.index, fill_value=0))

#     # ───────────────────────────────────────────────────────────────────────
#     #  Origin Counts
#     # ───────────────────────────────────────────────────────────────────────
#     #Count the number of cold emails in origin column and group by owner
#     cold_emails = all_stages[all_stages['origin'] == 'cold-email']
#     cold_emails_fdate = fdate(cold_emails, filter_column)
#     kpi["Cold Emails"] = cold_emails_fdate.groupby("Owner").size().reindex(
#         kpi.index, fill_value=0)

#     #Count the number of linkedin in origin column and group by owner
#     # linkedin = all_stages[all_stages['origin'] == 'LinkedIn']
#     # kpi["LinkedIn"] = linkedin.groupby("Owner").size().reindex(kpi.index,
#     #                                                            fill_value=0)
#     # kpi["LinkedIn"] = np.where(kpi['New Calls Booked'] > 0,
#     #                    kpi['New Calls Booked'] - kpi['Cold Emails'], 0)

#     # ───────────────────────────────────────────────────────────────────────
#     #  RATE METRICS  (all numeric, no NaN/None)
#     # ───────────────────────────────────────────────────────────────────────
#     with np.errstate(divide="ignore", invalid="ignore"):
#         kpi["Show Rate %"] = (kpi["Sales Call Taken"] / kpi["New Calls Booked"]
#                               ).replace([np.inf, -np.inf], 0).fillna(0) * 100

#         kpi["Unqualified Rate %"] = (kpi["Unqualified"] /
#                                      kpi["New Calls Booked"]).replace(
#                                          [np.inf, -np.inf], 0).fillna(0) * 100

#         kpi["Cancellation Rate %"] = (kpi["Cancelled Calls"] /
#                                       kpi["New Calls Booked"]).replace(
#                                           [np.inf, -np.inf], 0).fillna(0) * 100

#         kpi["Proposal Rate %"] = (kpi["Proposals"] /
#                                   kpi["New Calls Booked"]).replace(
#                                       [np.inf, -np.inf], 0).fillna(0) * 100

#     # Close metrics
#     closes = fdate(op_won, prop_date_col).groupby("Owner").size()
#     kpi["Close"] = closes.reindex(kpi.index, fill_value=0)  # helper column

#     with np.errstate(divide="ignore", invalid="ignore"):
#         kpi["Close Rate %"] = (kpi["Close"] / kpi["New Calls Booked"]).replace(
#             [np.inf, -np.inf], 0).fillna(0) * 100

#         kpi["Close Rate(Show) %"] = (kpi["Close"] /
#                                      kpi["Sales Call Taken"]).replace(
#                                          [np.inf, -np.inf], 0).fillna(0) * 100

#         kpi["Close Rate(MQL) %"] = (kpi["Close"] / kpi["Proposals"].replace(
#             0, np.nan)).replace([np.inf, -np.inf], 0).fillna(0) * 100

#     # ───────────────────────────────────────────────────────────────────────
#     #  REVENUE METRICS
#     # ───────────────────────────────────────────────────────────────────────
#     # Always use the user-selected filter_column for consistency
#     won_rev = fdate(op_won.copy(), filter_column)
#     won_rev["Deal Value"] = pd.to_numeric(won_rev["Deal Value"],
#                                           errors="coerce").fillna(0)

#     rev_sum = won_rev.groupby("Owner")["Deal Value"].sum()
#     kpi["Closed Revenue $"] = rev_sum.reindex(kpi.index, fill_value=0)

#     with np.errstate(divide="ignore", invalid="ignore"):
#         kpi["Revenue Per Call $"] = (kpi["Closed Revenue $"] /
#                                      kpi["New Calls Booked"]).replace(
#                                          [np.inf, -np.inf], 0).fillna(0)

#         kpi["Revenue Per Showed Up $"] = (kpi["Closed Revenue $"] /
#                                           kpi["Sales Call Taken"]).replace(
#                                               [np.inf, -np.inf], 0).fillna(0)

#         kpi["Revenue Per Proposal $"] = (
#             kpi["Closed Revenue $"] /
#             kpi["Proposals"].replace(0, np.nan)).replace([np.inf, -np.inf],
#                                                          0).fillna(0)

#     # Pipeline revenue (open proposals)
#     pipe_rev = fdate(op_proposal.copy(), prop_date_col)
#     pipe_rev["Deal Value"] = pd.to_numeric(pipe_rev["Deal Value"],
#                                            errors="coerce").fillna(0)
#     kpi["Pipeline Revenue $"] = (
#         pipe_rev.groupby("Owner")["Deal Value"].sum().reindex(kpi.index,
#                                                               fill_value=0))

#     # ── TOTAL ROW  ────────────────────────────────────────────────────
#     totals = {
#         "Owner": "Total",
#         "New Calls Booked": kpi["New Calls Booked"].sum(),
#         "Sales Call Taken": kpi["Sales Call Taken"].sum(),
#         "Proposals": kpi["Proposals"].sum(),
#         "Show Rate %": kpi["Show Rate %"].mean(),
#         "Unqualified": kpi["Unqualified"].sum(),
#         "Unqualified Rate %": kpi["Unqualified Rate %"].mean(),
#         "Cancelled Calls": kpi["Cancelled Calls"].sum(),
#         "Cancellation Rate %": kpi["Cancellation Rate %"].mean(),
#         "Proposal Rate %": kpi["Proposal Rate %"].mean(),
#         "Close Rate %": kpi["Close Rate %"].mean(),
#         "Close Rate(Show) %": kpi["Close Rate(Show) %"].mean(),
#         "Close Rate(MQL) %": kpi["Close Rate(MQL) %"].mean(),
#         "Closed Revenue $": kpi["Closed Revenue $"].sum(),
#         "Revenue Per Call $": kpi["Revenue Per Call $"].mean(),
#         "Revenue Per Showed Up $": kpi["Revenue Per Showed Up $"].mean(),
#         "Revenue Per Proposal $": kpi["Revenue Per Proposal $"].mean(),
#         "Pipeline Revenue $": kpi["Pipeline Revenue $"].sum(),
#         "Cold Emails": kpi["Cold Emails"].sum(),
#         #"LinkedIn": kpi["LinkedIn"].sum(),
#     }

#     # Create totals row with proper index
#     totals_df = pd.DataFrame([totals])

#     # Concatenate the main KPI data with totals
#     kpi_final = pd.concat([kpi.reset_index(drop=True), totals_df],
#                           ignore_index=True).drop(columns=["Close"],
#                                                   errors="ignore")
#     return kpi_final

def process_data_COLD_EMAIL(
    dataframes: dict[str, pd.DataFrame],
    st_date: str,
    end_date: str,
    filter_column: str,
) -> pd.DataFrame:
    """
    Build KPI table for COLD EMAIL leads in the range [st_date, end_date] (inclusive).

    Parameters
    ----------
    dataframes     Output of `fetch_data()`; keys like 'scheduled', 'won', …
    st_date        ISO date *YYYY-MM-DD* – range start
    end_date       ISO date *YYYY-MM-DD* – range end
    filter_column  Column chosen in the UI for date filtering
                   (e.g. 'Date Created', 'Sales Call Date')
    """
    start, end = map(lambda d: pd.to_datetime(d).date(), (st_date, end_date))

    # Stage dataframes
    stage = lambda k: dataframes.get(k, pd.DataFrame()).copy()
    op_cancelled, op_lost, op_noshow = map(stage, ("cancelled", "lost", "noshow"))
    op_proposal, op_scheduled = map(stage, ("proposal", "scheduled"))
    op_unqualified, op_won = map(stage, ("unqualified", "won"))

    all_stages = pd.concat(
        [
            op_cancelled,
            op_lost,
            op_noshow,
            op_proposal,
            op_scheduled,
            op_unqualified,
            op_won,
        ],
        copy=False,
    )

    owners = all_stages["Owner"].dropna().unique()
    kpi = pd.DataFrame(index=pd.Index(owners, name="Owner")).assign(Owner=owners)

    def _count(df: pd.DataFrame, date_col: str = filter_column) -> pd.Series:
        return (
            _filter_by_date(df, date_col, start, end)
            .groupby("Owner")
            .size()
            .reindex(kpi.index, fill_value=0)
        )

    kpi["New Calls Booked"] = _count(all_stages)
    
    # For metrics related to actual sales calls, use Sales Call Date if available, otherwise fall back to filter_column
    sales_call_date_col = "Sales Call Date" if "Sales Call Date" in all_stages.columns else filter_column
    
    kpi["Sales Call Taken"] = _count(
        pd.concat([op_unqualified, op_proposal, op_won, op_lost], copy=False), sales_call_date_col
    )
    kpi["Unqualified"] = _count(op_unqualified, "Sales Call Date" if "Sales Call Date" in all_stages.columns else filter_column)
    kpi["Cancelled Calls"] = _count(op_cancelled)

    prop_date_col = (
        "Sales Call Date" if "Sales Call Date" in op_proposal.columns else filter_column
    )
    kpi["Proposals"] = _count(
        pd.concat([op_proposal, op_won, op_lost], copy=False), prop_date_col
    )

    kpi['Won'] = _count(op_won, prop_date_col)

    # Origin: Cold Email only
    kpi["Cold Emails"] = _count(all_stages[all_stages["origin"] == "cold-email"])

    # Close count
    kpi["Close"] = (
        _filter_by_date(op_won, prop_date_col, start, end)
        .groupby("Owner")
        .size()
        .reindex(kpi.index, fill_value=0)
    )

    # Revenue: Closed + Pipeline
    def _sum_deal_value(df: pd.DataFrame) -> pd.Series:
        if df.empty:
            return pd.Series(0, index=kpi.index)
        df = df.copy()
        df["Deal Value"] = pd.to_numeric(df["Deal Value"], errors="coerce").fillna(0)
        return (
            _filter_by_date(df, filter_column, start, end)
            .groupby("Owner")["Deal Value"]
            .sum()
            .reindex(kpi.index, fill_value=0)
        )

    kpi["Closed Revenue $"] = _sum_deal_value(op_won)
    kpi["Pipeline Revenue $"] = _sum_deal_value(op_proposal)

    # Derived Metrics (same as process_data)
    for metric in DERIVED_METRICS:
        kpi[metric.name] = metric.producer(kpi)

    # Total row
    total_row = _build_total_row(kpi)
    kpi_final = (
        pd.concat([kpi, pd.DataFrame([total_row], index=["Total"])]).
        drop(columns=["Close"], errors="ignore").
        reset_index(drop=True)
    )

    return kpi_final


def process_data_GOOGLE_ADS_KPI(
    dataframes: dict[str, pd.DataFrame],
    st_date: str,
    end_date: str,
    filter_column: str,
) -> pd.DataFrame:
    """
    Build KPI table for GOOGLE ADS leads (UTM Source = 'google-ads') in the range [st_date, end_date] (inclusive).

    Parameters
    ----------
    dataframes     Output of `fetch_data()`; keys like 'scheduled', 'won', …
    st_date        ISO date *YYYY-MM-DD* – range start
    end_date       ISO date *YYYY-MM-DD* – range end
    filter_column  Column chosen in the UI for date filtering
                   (e.g. 'Date Created', 'Sales Call Date')
    """
    start, end = map(lambda d: pd.to_datetime(d).date(), (st_date, end_date))

    # Stage dataframes
    stage = lambda k: dataframes.get(k, pd.DataFrame()).copy()
    op_cancelled, op_lost, op_noshow = map(stage, ("cancelled", "lost", "noshow"))
    op_proposal, op_scheduled = map(stage, ("proposal", "scheduled"))
    op_unqualified, op_won = map(stage, ("unqualified", "won"))

    all_stages = pd.concat(
        [
            op_cancelled,
            op_lost,
            op_noshow,
            op_proposal,
            op_scheduled,
            op_unqualified,
            op_won,
        ],
        copy=False,
    )

    owners = all_stages["Owner"].dropna().unique()
    kpi = pd.DataFrame(index=pd.Index(owners, name="Owner")).assign(Owner=owners)

    def _count(df: pd.DataFrame, date_col: str = filter_column) -> pd.Series:
        return (
            _filter_by_date(df, date_col, start, end)
            .groupby("Owner")
            .size()
            .reindex(kpi.index, fill_value=0)
        )

    kpi["New Calls Booked"] = _count(all_stages)
    kpi["Sales Call Taken"] = _count(
        pd.concat([op_unqualified, op_proposal, op_won, op_lost], copy=False), "Sales Call Date" if "Sales Call Date" in all_stages.columns else filter_column
    )
    kpi["Unqualified"] = _count(op_unqualified, "Sales Call Date" if "Sales Call Date" in all_stages.columns else filter_column)
    kpi["Cancelled Calls"] = _count(op_cancelled)

    prop_date_col = (
        "Sales Call Date" if "Sales Call Date" in op_proposal.columns else filter_column
    )
    kpi["Proposals"] = _count(
        pd.concat([op_proposal, op_won, op_lost], copy=False), prop_date_col
    )

    kpi['Won'] = _count(op_won, prop_date_col)
    # Origin: Google Ads only
    kpi["Google Ads"] = _count(all_stages[all_stages["UTM Source"] == "google-ads"])

    # Close count
    kpi["Close"] = (
        _filter_by_date(op_won, prop_date_col, start, end)
        .groupby("Owner")
        .size()
        .reindex(kpi.index, fill_value=0)
    )

    # Revenue: Closed + Pipeline
    def _sum_deal_value(df: pd.DataFrame) -> pd.Series:
        if df.empty:
            return pd.Series(0, index=kpi.index)
        df = df.copy()
        df["Deal Value"] = pd.to_numeric(df["Deal Value"], errors="coerce").fillna(0)
        return (
            _filter_by_date(df, filter_column, start, end)
            .groupby("Owner")["Deal Value"]
            .sum()
            .reindex(kpi.index, fill_value=0)
        )

    kpi["Closed Revenue $"] = _sum_deal_value(op_won)
    kpi["Pipeline Revenue $"] = _sum_deal_value(op_proposal)

    # Derived Metrics (same as process_data)
    for metric in DERIVED_METRICS:
        kpi[metric.name] = metric.producer(kpi)

    # Total row
    total_row = _build_total_row(kpi)
    kpi_final = (
        pd.concat([kpi, pd.DataFrame([total_row], index=["Total"])]).
        drop(columns=["Close"], errors="ignore").
        reset_index(drop=True)
    )

    return kpi_final


def process_data_GOOGLE_ADS_KPI_BY_CAMPAIGN(
    dataframes: dict[str, pd.DataFrame],
    st_date: str,
    end_date: str,
    filter_column: str,
) -> pd.DataFrame:
    """
    Build KPI table for GOOGLE ADS leads grouped by UTM Campaign in the range [st_date, end_date] (inclusive).
    Uses only data where UTM Source = 'google-ads'.

    Parameters
    ----------
    dataframes     Output of `fetch_data()`; keys like 'scheduled', 'won', …
    st_date        ISO date *YYYY-MM-DD* – range start
    end_date       ISO date *YYYY-MM-DD* – range end
    filter_column  Column chosen in the UI for date filtering
                   (e.g. 'Date Created', 'Sales Call Date')
    """
    start, end = map(lambda d: pd.to_datetime(d).date(), (st_date, end_date))

    # Stage dataframes
    stage = lambda k: dataframes.get(k, pd.DataFrame()).copy()
    op_cancelled, op_lost, op_noshow = map(stage, ("cancelled", "lost", "noshow"))
    op_proposal, op_scheduled = map(stage, ("proposal", "scheduled"))
    op_unqualified, op_won = map(stage, ("unqualified", "won"))

    all_stages = pd.concat(
        [
            op_cancelled,
            op_lost,
            op_noshow,
            op_proposal,
            op_scheduled,
            op_unqualified,
            op_won,
        ],
        copy=False,
    )

    # Filter for google-ads only
    if "UTM Source" in all_stages.columns:
        all_stages = all_stages[all_stages["UTM Source"] == "google-ads"]

    if all_stages.empty or "UTM Campaign" not in all_stages.columns:
        return pd.DataFrame(columns=["UTM Campaign"])

    campaigns = all_stages["UTM Campaign"].dropna().unique()
    kpi = pd.DataFrame(index=pd.Index(campaigns, name="UTM Campaign")).assign(**{"UTM Campaign": campaigns})

    def _count(df: pd.DataFrame, date_col: str = filter_column) -> pd.Series:
        return (
            _filter_by_date(df, date_col, start, end)
            .groupby("UTM Campaign")
            .size()
            .reindex(kpi.index, fill_value=0)
        )

    kpi["New Calls Booked"] = _count(all_stages)
    
    sales_call_date_col = "Sales Call Date" if "Sales Call Date" in all_stages.columns else filter_column
    
    kpi["Sales Call Taken"] = _count(
        pd.concat([op_unqualified, op_proposal, op_won, op_lost], copy=False), sales_call_date_col
    )
    kpi["Unqualified"] = _count(op_unqualified, sales_call_date_col)
    kpi["Cancelled Calls"] = _count(op_cancelled)

    prop_date_col = (
        "Sales Call Date" if "Sales Call Date" in op_proposal.columns else filter_column
    )
    kpi["Proposals"] = _count(
        pd.concat([op_proposal, op_won, op_lost], copy=False), prop_date_col
    )
    kpi['Won'] = _count(op_won, prop_date_col)
    # Origin: Google Ads only
    kpi["Google Ads"] = _count(all_stages)

    # Close count
    kpi["Close"] = (
        _filter_by_date(op_won, prop_date_col, start, end)
        .groupby("UTM Campaign")
        .size()
        .reindex(kpi.index, fill_value=0)
    )

    # Revenue: Closed + Pipeline
    def _sum_deal_value(df: pd.DataFrame) -> pd.Series:
        if df.empty:
            return pd.Series(0, index=kpi.index)
        df = df.copy()
        df["Deal Value"] = pd.to_numeric(df["Deal Value"], errors="coerce").fillna(0)
        return (
            _filter_by_date(df, filter_column, start, end)
            .groupby("UTM Campaign")["Deal Value"]
            .sum()
            .reindex(kpi.index, fill_value=0)
        )

    kpi["Closed Revenue $"] = _sum_deal_value(op_won)
    kpi["Pipeline Revenue $"] = _sum_deal_value(op_proposal)

    # Derived Metrics (same as process_data)
    for metric in DERIVED_METRICS:
        kpi[metric.name] = metric.producer(kpi)

    # Total row - calculate for all campaigns combined
    def _build_total_row_campaign(df: pd.DataFrame) -> dict[str, float | str]:
        totals: dict[str, float | str] = {"UTM Campaign": "Total"}
        for col in df.columns:
            if col == "UTM Campaign":
                continue
            if not is_numeric_dtype(df[col]):
                continue
            totals[col] = df[col].sum() if col in RAW_METRICS else df[col].mean()
        return totals

    total_row = _build_total_row_campaign(kpi)
    kpi_final = (
        pd.concat([kpi, pd.DataFrame([total_row], index=["Total"])]).
        drop(columns=["Close"], errors="ignore").
        reset_index(drop=True)
    )

    return kpi_final


def process_data_GOOGLE_ADS_KPI_BY_CONTENT(
    dataframes: dict[str, pd.DataFrame],
    st_date: str,
    end_date: str,
    filter_column: str,
) -> pd.DataFrame:
    """
    Build KPI table for GOOGLE ADS leads grouped by UTM Content in the range [st_date, end_date] (inclusive).
    Uses only data where UTM Source = 'google-ads'.

    Parameters
    ----------
    dataframes     Output of `fetch_data()`; keys like 'scheduled', 'won', …
    st_date        ISO date *YYYY-MM-DD* – range start
    end_date       ISO date *YYYY-MM-DD* – range end
    filter_column  Column chosen in the UI for date filtering
                   (e.g. 'Date Created', 'Sales Call Date')
    """
    start, end = map(lambda d: pd.to_datetime(d).date(), (st_date, end_date))

    # Stage dataframes
    stage = lambda k: dataframes.get(k, pd.DataFrame()).copy()
    op_cancelled, op_lost, op_noshow = map(stage, ("cancelled", "lost", "noshow"))
    op_proposal, op_scheduled = map(stage, ("proposal", "scheduled"))
    op_unqualified, op_won = map(stage, ("unqualified", "won"))

    all_stages = pd.concat(
        [
            op_cancelled,
            op_lost,
            op_noshow,
            op_proposal,
            op_scheduled,
            op_unqualified,
            op_won,
        ],
        copy=False,
    )

    # Filter for google-ads only
    if "UTM Source" in all_stages.columns:
        all_stages = all_stages[all_stages["UTM Source"] == "google-ads"]

    if all_stages.empty or "UTM Content" not in all_stages.columns:
        return pd.DataFrame(columns=["UTM Content"])

    contents = all_stages["UTM Content"].dropna().unique()
    kpi = pd.DataFrame(index=pd.Index(contents, name="UTM Content")).assign(**{"UTM Content": contents})

    def _count(df: pd.DataFrame, date_col: str = filter_column) -> pd.Series:
        return (
            _filter_by_date(df, date_col, start, end)
            .groupby("UTM Content")
            .size()
            .reindex(kpi.index, fill_value=0)
        )

    kpi["New Calls Booked"] = _count(all_stages)
    
    sales_call_date_col = "Sales Call Date" if "Sales Call Date" in all_stages.columns else filter_column
    
    kpi["Sales Call Taken"] = _count(
        pd.concat([op_unqualified, op_proposal, op_won, op_lost], copy=False), sales_call_date_col
    )
    kpi["Unqualified"] = _count(op_unqualified, sales_call_date_col)
    kpi["Cancelled Calls"] = _count(op_cancelled)

    prop_date_col = (
        "Sales Call Date" if "Sales Call Date" in op_proposal.columns else filter_column
    )
    kpi["Proposals"] = _count(
        pd.concat([op_proposal, op_won, op_lost], copy=False), prop_date_col
    )
    kpi['Won'] = _count(op_won, prop_date_col)
    # Origin: Google Ads only
    kpi["Google Ads"] = _count(all_stages)

    # Close count
    kpi["Close"] = (
        _filter_by_date(op_won, prop_date_col, start, end)
        .groupby("UTM Content")
        .size()
        .reindex(kpi.index, fill_value=0)
    )

    # Revenue: Closed + Pipeline
    def _sum_deal_value(df: pd.DataFrame) -> pd.Series:
        if df.empty:
            return pd.Series(0, index=kpi.index)
        df = df.copy()
        df["Deal Value"] = pd.to_numeric(df["Deal Value"], errors="coerce").fillna(0)
        return (
            _filter_by_date(df, filter_column, start, end)
            .groupby("UTM Content")["Deal Value"]
            .sum()
            .reindex(kpi.index, fill_value=0)
        )

    kpi["Closed Revenue $"] = _sum_deal_value(op_won)
    kpi["Pipeline Revenue $"] = _sum_deal_value(op_proposal)

    # Derived Metrics (same as process_data)
    for metric in DERIVED_METRICS:
        kpi[metric.name] = metric.producer(kpi)

    # Total row - calculate for all content combined
    def _build_total_row_content(df: pd.DataFrame) -> dict[str, float | str]:
        totals: dict[str, float | str] = {"UTM Content": "Total"}
        for col in df.columns:
            if col == "UTM Content":
                continue
            if not is_numeric_dtype(df[col]):
                continue
            totals[col] = df[col].sum() if col in RAW_METRICS else df[col].mean()
        return totals

    total_row = _build_total_row_content(kpi)
    kpi_final = (
        pd.concat([kpi, pd.DataFrame([total_row], index=["Total"])]).
        drop(columns=["Close"], errors="ignore").
        reset_index(drop=True)
    )

    return kpi_final


def process_data_Google_Ads(
    dataframes: dict[str, pd.DataFrame],
    st_date: str,
    end_date: str,
    filter_column: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Return two DataFrames aggregated by Google-Ads UTM parameters.

    Returns
    -------
    df_campaigns : pd.DataFrame
        Columns → ['UTM Campaign', 'Count', 'Total Deal Value']
    df_content   : pd.DataFrame
        Columns → ['UTM Content',  'Count', 'Total Deal Value']
    """

    start, end = map(lambda d: pd.to_datetime(d).date(), (st_date, end_date))

    # combine all pipeline stages
    stage = lambda k: dataframes.get(k, pd.DataFrame())
    all_stages = pd.concat(
        [
            stage("cancelled"),
            stage("lost"),
            stage("noshow"),
            stage("proposal"),
            stage("scheduled"),
            stage("unqualified"),
            stage("won"),
        ],
        copy=False,
    )

    # short-circuit if nothing
    if all_stages.empty:
        empty = lambda key: pd.DataFrame(columns=[key, "Count", "Total Deal Value"])
        return empty("UTM Campaign"), empty("UTM Content")

    # date filter
    if filter_column in all_stages.columns:
        all_stages = _filter_by_date(all_stages, filter_column, start, end)

    # numeric Deal Value
    if "Deal Value" not in all_stages.columns:
        all_stages["Deal Value"] = 0.0
    all_stages["Deal Value"] = pd.to_numeric(all_stages["Deal Value"], errors="coerce").fillna(0)

    # generic aggregator
    def _aggregate(df: pd.DataFrame, key: str) -> pd.DataFrame:
        if key not in df.columns or df.empty:
            return pd.DataFrame(columns=[key, "Count", "Total Deal Value"])
        grouped = (
            df.groupby(key)
            .agg(Count=(key, "size"), **{"Total Deal Value": ("Deal Value", "sum")})
            .reset_index()
        )
        return grouped

    df_campaigns = _aggregate(all_stages, "UTM Campaign")
    df_content   = _aggregate(all_stages, "UTM Content")

    return df_campaigns, df_content


# ──────────────────────────────  Helpers  ──────────────────────────────── #
def _coerce_date_series(series: pd.Series) -> pd.Series:
    """Vectorised →date conversion (returns `datetime.date` or `NaT`)."""
    return pd.to_datetime(series, format='mixed', errors="coerce").dt.date


def _filter_by_date(
    df: pd.DataFrame, date_col: str, start: date, end: date
) -> pd.DataFrame:
    """Inclusive date-range filter; keeps schema on missing column."""
    if date_col not in df.columns:
        return df.iloc[0:0]  # empty with same columns
    dates = _coerce_date_series(df[date_col])
    mask = (dates >= start) & (dates <= end)
    return df.loc[mask]


def _safe_divide(num: pd.Series | float, den: pd.Series | float) -> pd.Series:
    """Element-wise division → 0 where denom is 0/NaN/inf."""
    return (num / den.replace(0, np.nan)).fillna(0.0).replace([np.inf, -np.inf], 0.0)


# ───────────────────────  Declarative metric registry  ─────────────────── #
@dataclass(frozen=True)
class Metric:
    """Derived metric definition."""
    name: str
    producer: Callable[[pd.DataFrame], pd.Series]


def _rate(numer: str, denom: str) -> Callable[[pd.DataFrame], pd.Series]:
    return lambda k: _safe_divide(k[numer], k[denom]) * 100.0


def _revenue_per(base: str) -> Callable[[pd.DataFrame], pd.Series]:
    return lambda k: _safe_divide(k["Closed Revenue $"], k[base])


RAW_METRICS: Final[tuple[str, ...]] = (
    "New Calls Booked",
    "Sales Call Taken",
    "Unqualified",
    "Cancelled Calls",
    "Proposals",
    "Won",
    "Cold Emails",
    "LinkedIn",
    "Closed Revenue $",
    "Pipeline Revenue $",
)

DERIVED_METRICS: Final[tuple[Metric, ...]] = (
    Metric("Show Rate %", _rate("Sales Call Taken", "New Calls Booked")),
    Metric("Unqualified Rate %", _rate("Unqualified", "New Calls Booked")),
    Metric("Cancellation Rate %", _rate("Cancelled Calls", "New Calls Booked")),
    Metric("Proposal Rate %", _rate("Proposals", "New Calls Booked")),
    Metric("Close Rate %", _rate("Close", "New Calls Booked")),
    Metric("Close Rate(Show) %", _rate("Close", "Sales Call Taken")),
    Metric("Close Rate(MQL) %", _rate("Close", "Proposals")),
    Metric("Revenue Per Call $", _revenue_per("New Calls Booked")),
    Metric("Revenue Per Showed Up $", _revenue_per("Sales Call Taken")),
    Metric("Revenue Per Proposal $", _revenue_per("Proposals")),
)


# ────────────────────────  KPI total-row helper  ───────────────────────── #
def _build_total_row(df: pd.DataFrame) -> dict[str, float | str]:
    """Aggregate sums/means for numeric columns; skip object columns."""
    totals: dict[str, float | str] = {"Owner": "Total"}
    for col in df.columns:
        if col == "Owner":  # already set
            continue
        if not is_numeric_dtype(df[col]):
            continue  # ignore non-numeric
        totals[col] = df[col].sum() if col in RAW_METRICS else df[col].mean()
    return totals


# ───────────────────────────  Main function  ───────────────────────────── #
def process_data(
    dataframes: dict[str, pd.DataFrame],
    st_date: str,
    end_date: str,
    filter_column: str,
) -> pd.DataFrame:
    """
    Build the KPI table for the inclusive date range [st_date, end_date].

    Parameters
    ----------
    dataframes     Output of `fetch_data()`; keys like 'scheduled', 'won', …
    st_date        ISO date *YYYY-MM-DD* – range start
    end_date       ISO date *YYYY-MM-DD* – range end
    filter_column  Column chosen in the UI for date filtering
                   (e.g. 'Date Created', 'Sales Call Date', …)
    """
    # ───────────────  Preparations  ─────────────── #
    start, end = map(lambda d: pd.to_datetime(d).date(), (st_date, end_date))

    # Shorthand to fetch a copy or empty DF
    stage = lambda k: dataframes.get(k, pd.DataFrame()).copy()
    op_cancelled, op_lost, op_noshow = map(stage, ("cancelled", "lost", "noshow"))
    op_proposal, op_scheduled = map(stage, ("proposal", "scheduled"))
    op_unqualified, op_won = map(stage, ("unqualified", "won"))

    owners = (
        pd.concat(
            [
                op_cancelled,
                op_lost,
                op_noshow,
                op_proposal,
                op_scheduled,
                op_unqualified,
                op_won,
            ],
            copy=False,
        )["Owner"]
        .dropna()
        .unique()
    )
    kpi = pd.DataFrame(index=pd.Index(owners, name="Owner")).assign(Owner=owners)

    # ───────────────  Raw counts  ─────────────── #
    def _count(df: pd.DataFrame, date_col: str = filter_column) -> pd.Series:
        return (
            _filter_by_date(df, date_col, start, end)
            .groupby("Owner")
            .size()
            .reindex(kpi.index, fill_value=0)
        )

    all_stages = pd.concat(
        [
            op_cancelled,
            op_lost,
            op_noshow,
            op_proposal,
            op_scheduled,
            op_unqualified,
            op_won,
        ],
        copy=False,
    )

    kpi["New Calls Booked"] = _count(all_stages)
    kpi["Sales Call Taken"] = _count(
        pd.concat([op_unqualified, op_proposal, op_won, op_lost], copy=False)
    )
    kpi["Unqualified"] = _count(op_unqualified)
    kpi["Cancelled Calls"] = _count(op_cancelled)

    prop_date_col = (
        "Sales Call Date" if "Sales Call Date" in op_proposal.columns else filter_column
    )
    kpi["Proposals"] = _count(
        pd.concat([op_proposal, op_won, op_lost], copy=False), prop_date_col
    )
    kpi['Won'] = _count(op_won, prop_date_col)
    # Origination
    kpi["Cold Emails"] = _count(all_stages[all_stages["origin"] == "cold-email"])
    kpi["LinkedIn"] = np.where(
        kpi["New Calls Booked"] > 0,
        kpi["New Calls Booked"] - kpi["Cold Emails"],
        0,
    )

    # ───────────────  Close + revenue  ─────────────── #
    kpi["Close"] = (
        _filter_by_date(op_won, prop_date_col, start, end)
        .groupby("Owner")
        .size()
        .reindex(kpi.index, fill_value=0)
    )

    def _sum_deal_value(df: pd.DataFrame) -> pd.Series:
        if df.empty:
            return pd.Series(0, index=kpi.index)
        df = df.copy()
        df["Deal Value"] = pd.to_numeric(df["Deal Value"], errors="coerce").fillna(0)
        return (
            _filter_by_date(df, filter_column, start, end)
            .groupby("Owner")["Deal Value"]
            .sum()
            .reindex(kpi.index, fill_value=0)
        )

    kpi["Closed Revenue $"] = _sum_deal_value(op_won)
    kpi["Pipeline Revenue $"] = _sum_deal_value(op_proposal)

    # ───────────────  Derived metrics  ─────────────── #
    for metric in DERIVED_METRICS:
        kpi[metric.name] = metric.producer(kpi)

    # ───────────────  Total row  ─────────────── #
    kpi_final = (
        pd.concat([kpi, pd.DataFrame([_build_total_row(kpi)], index=["Total"])])
        .drop(columns=["Close"], errors="ignore")
        .reset_index(drop=True)
    )

    return kpi_final


def process_data_LINKEDIN(
    dataframes: dict[str, pd.DataFrame],
    st_date: str,
    end_date: str,
    filter_column: str,
) -> pd.DataFrame:
    """
    Build KPI table for LINKEDIN leads in the range [st_date, end_date] (inclusive).

    Parameters
    ----------
    dataframes     Output of `fetch_data()`; keys like 'scheduled', 'won', …
    st_date        ISO date *YYYY-MM-DD* – range start
    end_date       ISO date *YYYY-MM-DD* – range end
    filter_column  Column chosen in the UI for date filtering
                   (e.g. 'Date Created', 'Sales Call Date')
    """
    start, end = map(lambda d: pd.to_datetime(d).date(), (st_date, end_date))

    # Stage dataframes
    stage = lambda k: dataframes.get(k, pd.DataFrame()).copy()
    op_cancelled, op_lost, op_noshow = map(stage, ("cancelled", "lost", "noshow"))
    op_proposal, op_scheduled = map(stage, ("proposal", "scheduled"))
    op_unqualified, op_won = map(stage, ("unqualified", "won"))

    all_stages = pd.concat(
        [
            op_cancelled,
            op_lost,
            op_noshow,
            op_proposal,
            op_scheduled,
            op_unqualified,
            op_won,
        ],
        copy=False,
    )

    owners = all_stages["Owner"].dropna().unique()
    kpi = pd.DataFrame(index=pd.Index(owners, name="Owner")).assign(Owner=owners)

    def _count(df: pd.DataFrame, date_col: str = filter_column) -> pd.Series:
        return (
            _filter_by_date(df, date_col, start, end)
            .groupby("Owner")
            .size()
            .reindex(kpi.index, fill_value=0)
        )

    kpi["New Calls Booked"] = _count(all_stages)
    kpi["Sales Call Taken"] = _count(
        pd.concat([op_unqualified, op_proposal, op_won, op_lost], copy=False), "Sales Call Date" if "Sales Call Date" in all_stages.columns else filter_column
    )
    kpi["Unqualified"] = _count(op_unqualified)
    kpi["Cancelled Calls"] = _count(op_cancelled)

    prop_date_col = (
        "Sales Call Date" if "Sales Call Date" in op_proposal.columns else filter_column
    )
    kpi["Proposals"] = _count(
        pd.concat([op_proposal, op_won, op_lost], copy=False), prop_date_col
    )
    kpi['Won'] = _count(op_won, prop_date_col)
    # Origin: Linkedin only
    kpi["LINKEDIN"] = _count(all_stages[all_stages["origin"] == "LinkedIn"])

    # Close count
    kpi["Close"] = (
        _filter_by_date(op_won, prop_date_col, start, end)
        .groupby("Owner")
        .size()
        .reindex(kpi.index, fill_value=0)
    )

    # Revenue: Closed + Pipeline
    def _sum_deal_value(df: pd.DataFrame) -> pd.Series:
        if df.empty:
            return pd.Series(0, index=kpi.index)
        df = df.copy()
        df["Deal Value"] = pd.to_numeric(df["Deal Value"], errors="coerce").fillna(0)
        return (
            _filter_by_date(df, filter_column, start, end)
            .groupby("Owner")["Deal Value"]
            .sum()
            .reindex(kpi.index, fill_value=0)
        )

    kpi["Closed Revenue $"] = _sum_deal_value(op_won)
    kpi["Pipeline Revenue $"] = _sum_deal_value(op_proposal)

    # Derived Metrics (same as process_data)
    for metric in DERIVED_METRICS:
        kpi[metric.name] = metric.producer(kpi)

    # Total row
    total_row = _build_total_row(kpi)
    kpi_final = (
        pd.concat([kpi, pd.DataFrame([total_row], index=["Total"])]).
        drop(columns=["Close"], errors="ignore").
        reset_index(drop=True)
    )

    return kpi_final

