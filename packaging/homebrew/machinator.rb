class Machinator < Formula
  include Language::Python::Virtualenv

  desc "Prompt-first control-plane CLI for ML workspaces and pipelines"
  homepage "https://github.com/jmcguigan10/Machinator"
  url "https://github.com/jmcguigan10/Machinator/releases/download/v0.6.0/machinator-0.6.0.tar.gz"
  sha256 "5a271cc1ab7792c154218256505b847e993a941df365cadc36cff21f78428484"
  license "MIT"

  depends_on "python@3.12"

  def install
    virtualenv_install_with_resources
  end

  test do
    assert_match "Machinator", shell_output("#{bin}/macht --help")
  end
end
