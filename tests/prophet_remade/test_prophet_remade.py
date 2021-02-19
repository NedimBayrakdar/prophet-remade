#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for `prophet_remade` package."""

import pytest

import prophet_remade


@pytest.fixture
def response():
    """Sample pytest fixture.

    See more at: http://doc.pytest.org/en/latest/fixture.html
    """

    return "some_response"


def test_content(response):
    """Sample pytest test function with the pytest fixture as an argument."""

    assert response


def test_module():
    """Dummy test that asserts prophet_remade was imported."""

    assert prophet_remade
